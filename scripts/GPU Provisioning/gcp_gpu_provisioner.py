# credit to edstem post
import argparse
import json
import logging
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from google.cloud import compute_v1
from tqdm import tqdm


# Configuration
@dataclass
class VMConfig:
    """VM configuration parameters"""
    machine_type: str = "g2-standard-8"
    disk_size_gb: int = 500
    disk_type: str = "pd-balanced"
    boot_image: str = "projects/ml-images/global/images/c0-deeplearning-common-cu124-v20241224-debian-11-py310"
    gpu_count: int = 1
    use_spot: bool = False


class GCPGPUProvisioner:
    """Handles GPU VM provisioning on Google Cloud Platform"""
    
    def __init__(self, project_id: str, config: VMConfig = None):
        """
        Initialize the provisioner.
        
        Args:
            project_id: GCP project ID
            config: VM configuration (uses defaults if not provided)
        """
        self.project_id = project_id
        self.config = config or VMConfig()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging with both file and console output"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler('gcp_gpu_provisioner.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def check_authentication(self) -> bool:
        """
        Verify that GCP authentication is properly configured.
        
        Returns:
            True if authenticated, False otherwise
        """
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                self.logger.info(f"Authenticated as: {result.stdout.strip()}")
                return True
            else:
                self.logger.error("No active GCP authentication found")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Authentication check failed: {e}")
            return False
        except FileNotFoundError:
            self.logger.error("gcloud CLI not found. Please install Google Cloud SDK")
            return False
    
    def authenticate(self) -> bool:
        """
        Authenticate with GCP using application default credentials.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            self.logger.info("Starting GCP authentication...")
            
            # Application default login
            subprocess.run(
                ['gcloud', 'auth', 'application-default', 'login'],
                check=True
            )
            
            # Set quota project
            subprocess.run(
                ['gcloud', 'auth', 'application-default', 'set-quota-project', self.project_id],
                check=True
            )
            
            # User login
            subprocess.run(
                ['gcloud', 'auth', 'login'],
                check=True
            )
            
            # Set project
            subprocess.run(
                ['gcloud', 'config', 'set', 'project', self.project_id],
                check=True
            )
            
            self.logger.info("Authentication successful")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def get_all_zones(self) -> List[str]:
        """
        Retrieve all available GCP zones for the project.
        
        Returns:
            List of zone names
        """
        self.logger.info("Fetching all available zones...")
        try:
            zones_client = compute_v1.ZonesClient()
            zones = [zone.name for zone in zones_client.list(project=self.project_id)]
            self.logger.info(f"Found {len(zones)} zones")
            return zones
        except Exception as e:
            self.logger.error(f"Failed to fetch zones: {e}")
            return []
    
    def list_all_gpu_types(self, zones: List[str] = None) -> pd.DataFrame:
        """
        List all available GPU types across all zones.
        
        Args:
            zones: List of zones to check (fetches all if not provided)
            
        Returns:
            DataFrame with columns: gpu_type, zone, description
        """
        if zones is None:
            zones = self.get_all_zones()
        
        self.logger.info("Scanning zones for GPU types...")
        results = []
        
        accelerator_client = compute_v1.AcceleratorTypesClient()
        
        for zone in tqdm(zones, desc="Checking zones"):
            try:
                request = compute_v1.ListAcceleratorTypesRequest(
                    project=self.project_id,
                    zone=zone
                )
                responses = accelerator_client.list(request=request)
                
                for response in responses:
                    results.append({
                        "gpu_type": response.name,
                        "zone": zone,
                        "description": response.description,
                    })
            except Exception as e:
                self.logger.debug(f"Error checking zone {zone}: {e}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.drop_duplicates().sort_values(by=["gpu_type", "zone"]).reset_index(drop=True)
            self.logger.info(f"Found {len(df)} GPU type/zone combinations")
        else:
            self.logger.warning("No GPU types found in any zone")
        
        return df
    
    def find_zones_with_gpu(self, gpu_type: str, zones: List[str] = None) -> List[Dict[str, str]]:
        """
        Find all zones that have a specific GPU type available.
        
        Args:
            gpu_type: GPU type to search for (e.g., 'nvidia-l4')
            zones: List of zones to check (fetches all if not provided)
            
        Returns:
            List of dicts with keys: region, zone, gpu_type
        """
        if zones is None:
            zones = self.get_all_zones()
        
        self.logger.info(f"Searching for {gpu_type} across {len(zones)} zones...")
        results = []
        
        accelerator_client = compute_v1.AcceleratorTypesClient()
        
        for zone in tqdm(zones, desc=f"Searching for {gpu_type}"):
            try:
                request = compute_v1.ListAcceleratorTypesRequest(
                    project=self.project_id,
                    zone=zone
                )
                responses = accelerator_client.list(request=request)
                
                for response in responses:
                    if response.name == gpu_type:
                        region = "-".join(zone.split("-")[:-1])
                        results.append({
                            "region": region,
                            "zone": zone,
                            "gpu_type": response.name,
                        })
                        break
                        
            except Exception as e:
                self.logger.debug(f"Error checking zone {zone}: {e}")
                continue
        
        self.logger.info(f"Found {gpu_type} in {len(results)} zones")
        return results
    
    def _configure_disk(self, zone: str) -> compute_v1.AttachedDisk:
        """Configure boot disk with given specifications"""
        return compute_v1.AttachedDisk(
            auto_delete=True,
            boot=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image=self.config.boot_image,
                disk_size_gb=self.config.disk_size_gb,
                disk_type=f"projects/{self.project_id}/zones/{zone}/diskTypes/{self.config.disk_type}"
            )
        )
    
    def _configure_network(self, region: str) -> compute_v1.NetworkInterface:
        """Configure networking for the instance"""
        net_interface = compute_v1.NetworkInterface()
        # Remove external IP - will use Cloud NAT instead
        net_interface.stack_type = "IPV4_ONLY"
        net_interface.subnetwork = f"projects/{self.project_id}/regions/{region}/subnetworks/default"
        return net_interface
    
    def _configure_gpu(self, zone: str, gpu_type: str) -> compute_v1.AcceleratorConfig:
        """Configure GPU accelerator"""
        return compute_v1.AcceleratorConfig(
            accelerator_count=self.config.gpu_count,
            accelerator_type=f"projects/{self.project_id}/zones/{zone}/acceleratorTypes/{gpu_type}"
        )
    
    def _build_instance(
        self,
        name: str,
        zone: str,
        disk: compute_v1.AttachedDisk,
        gpu: compute_v1.AcceleratorConfig,
        network: compute_v1.NetworkInterface
    ) -> compute_v1.Instance:
        """Construct a VM instance configuration"""
        return compute_v1.Instance(
            name=name,
            machine_type=f"projects/{self.project_id}/zones/{zone}/machineTypes/{self.config.machine_type}",
            guest_accelerators=[gpu],
            scheduling=compute_v1.Scheduling(
                provisioning_model="SPOT" if self.config.use_spot else "STANDARD",
                automatic_restart=False if self.config.use_spot else True,
                on_host_maintenance="TERMINATE"
            ),
            disks=[disk],
            network_interfaces=[network],
        )
    
    def list_deep_learning_images(self, project="ml-images"):
        """List available deep learning images"""
        images_client = compute_v1.ImagesClient()
        images = []
        
        for image in images_client.list(project=project):
            images.append({
                "name": image.name,
                "family": image.family if hasattr(image, 'family') else "N/A",
                "description": image.description
            })
    
        return pd.DataFrame(images)
    
    def provision_vm(
        self,
        region: str,
        zone: str,
        instance_name: str,
        gpu_type: str,
        timeout: int = 1000
    ) -> tuple[bool, Optional[str]]:
        """
        Provision a single VM with GPU.
        
        Args:
            region: GCP region
            zone: GCP zone
            instance_name: Name for the VM instance
            gpu_type: GPU type to attach
            timeout: Operation timeout in seconds
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            disk = self._configure_disk(zone)
            network = self._configure_network(region)
            gpu = self._configure_gpu(zone, gpu_type)
            instance = self._build_instance(instance_name, zone, disk, gpu, network)
            
            instance_client = compute_v1.InstancesClient()
            operation = instance_client.insert(
                request=compute_v1.InsertInstanceRequest(
                    zone=zone,
                    project=self.project_id,
                    instance_resource=instance
                )
            )
            
            operation.result(timeout=timeout)
            self.logger.info(f"✓ Successfully created {instance_name} in {zone}")
            return True, None
            
        except Exception as e:
            error_msg = str(e)
            self.logger.debug(f"✗ Failed to create {instance_name} in {zone}: {error_msg}")
            return False, error_msg
    
    def provision_vms_batch(
        self,
        zone_list: List[Dict[str, str]],
        instance_prefix: str = "gpu-vm"
    ) -> pd.DataFrame:
        """
        Provision VMs across multiple zones and track results.
        
        Args:
            zone_list: List of dicts with keys: region, zone, gpu_type
            instance_prefix: Prefix for instance names
            
        Returns:
            DataFrame with results for each zone attempt
        """
        self.logger.info(f"Attempting to provision VMs in {len(zone_list)} zones...")
        results = []
        
        for entry in tqdm(zone_list, desc="Provisioning VMs"):
            unique_name = f"{instance_prefix}-{uuid.uuid4().hex[:8]}"
            region = entry["region"]
            zone = entry["zone"]
            gpu_type = entry["gpu_type"]
            
            start_time = time.time()
            success, error = self.provision_vm(region, zone, unique_name, gpu_type)
            elapsed_time = round(time.time() - start_time, 3) if success else "N/A"
            
            results.append({
                "Instance Name": unique_name if success else "N/A",
                "Region": region,
                "Zone": zone,
                "GPU Type": gpu_type,
                "GPU Allocated": 'Yes' if success else 'No',
                "Failure Reason": "N/A" if success else error,
                "Time Taken(s)": elapsed_time
            })
        
        df = pd.DataFrame(results)
        
        success_count = df[df['GPU Allocated'] == 'Yes'].shape[0]
        self.logger.info(f"Provisioning complete: {success_count}/{len(zone_list)} successful")
        
        return df
    
    def delete_vm(self, zone: str, instance_name: str) -> bool:
        """
        Delete a VM instance.
        
        Args:
            zone: GCP zone where the instance is located
            instance_name: Name of the instance to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            instance_client = compute_v1.InstancesClient()
            operation = instance_client.delete(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )
            operation.result()
            self.logger.info(f"✓ VM {instance_name} in {zone} deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error deleting {instance_name} in {zone}: {e}")
            return False
    
    def list_running_vms(self) -> pd.DataFrame:
        """
        List all running VM instances in the project.
        
        Returns:
            DataFrame with instance details
        """
        self.logger.info("Fetching running VMs...")
        instances = []
        
        try:
            instance_client = compute_v1.InstancesClient()
            
            # Get all zones
            zones = self.get_all_zones()
            
            for zone in tqdm(zones, desc="Checking zones"):
                try:
                    request = compute_v1.ListInstancesRequest(
                        project=self.project_id,
                        zone=zone
                    )
                    
                    for instance in instance_client.list(request=request):
                        gpu_info = []
                        if instance.guest_accelerators:
                            for gpu in instance.guest_accelerators:
                                gpu_type = gpu.accelerator_type.split('/')[-1]
                                gpu_info.append(f"{gpu_type} x{gpu.accelerator_count}")
                        
                        instances.append({
                            "Name": instance.name,
                            "Zone": zone,
                            "Status": instance.status,
                            "Machine Type": instance.machine_type.split('/')[-1],
                            "GPUs": ', '.join(gpu_info) if gpu_info else "None"
                        })
                        
                except Exception as e:
                    self.logger.debug(f"Error listing instances in {zone}: {e}")
                    continue
            
            df = pd.DataFrame(instances)
            self.logger.info(f"Found {len(df)} running VMs")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to list VMs: {e}")
            return pd.DataFrame()


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="GCP GPU Provisioning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file for all settings
  %(prog)s --config config.json --provision
  
  # Use config file but override GPU type
  %(prog)s --config config.json --gpu-type nvidia-l4 --list-zones
  
  # List all zones with nvidia-l4 GPUs (without config file)
  %(prog)s --project my-project --gpu-type nvidia-l4 --list-zones
  
  # Provision VMs with nvidia-l4 in available zones
  %(prog)s --project my-project --gpu-type nvidia-l4 --provision
  
  # List all GPU types available
  %(prog)s --project my-project --list-gpu-types
  
  # Delete a specific VM
  %(prog)s --project my-project --delete-vm vm-name --zone us-central1-a
  
  # List running VMs
  %(prog)s --project my-project --list-vms
        """
    )
    
    # Configuration file argument
    parser.add_argument(
        '--config',
        help='Path to JSON configuration file'
    )
    
    # Required arguments
    parser.add_argument(
        '--project',
        help='GCP project ID (required unless provided in config file)'
    )
    
    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--list-zones',
        action='store_true',
        help='List zones that have the specified GPU type'
    )
    action_group.add_argument(
        '--list-gpu-types',
        action='store_true',
        help='List all available GPU types across all zones'
    )
    action_group.add_argument(
        '--provision',
        action='store_true',
        help='Provision VMs with GPUs in available zones'
    )
    action_group.add_argument(
        '--delete-vm',
        metavar='VM_NAME',
        help='Delete a specific VM instance'
    )
    action_group.add_argument(
        '--list-vms',
        action='store_true',
        help='List all running VM instances'
    )
    action_group.add_argument(
        '--authenticate',
        action='store_true',
        help='Run GCP authentication flow'
    )
    action_group.add_argument('--list-images', action='store_true', help='List deep learning images')
    
    # Optional arguments
    parser.add_argument(
        '--gpu-type',
        help='GPU type to search for (overrides config file)'
    )
    parser.add_argument(
        '--zone',
        help='Specific zone (required for --delete-vm)'
    )
    parser.add_argument(
        '--machine-type',
        help='Machine type for VMs (overrides config file)'
    )
    parser.add_argument(
        '--disk-size',
        type=int,
        help='Boot disk size in GB (overrides config file)'
    )
    parser.add_argument(
        '--spot',
        action='store_true',
        help='Use spot instances (preemptible)'
    )
    parser.add_argument(
        '--output',
        help='Output file for results (CSV format)'
    )
    parser.add_argument(
        '--max-zones',
        type=int,
        help='Maximum number of zones to attempt provisioning (overrides config file)'
    )
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config_data = {}
    if args.config:
        try:
            with open(args.config) as f:
                config_data = json.load(f)
            print(f"✓ Loaded configuration from {args.config}")
        except FileNotFoundError:
            print(f"✗ Error: Config file not found: {args.config}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON in config file: {e}")
            sys.exit(1)
    
    # Merge config file with command-line arguments (CLI takes precedence)
    project_id = args.project or config_data.get('project_id')
    gpu_type = args.gpu_type or config_data.get('gpu_type', 'nvidia-l4')
    
    # Get VM config values
    vm_config = config_data.get('vm_config', {})
    machine_type = args.machine_type or vm_config.get('machine_type', 'g2-standard-8')
    disk_size_gb = args.disk_size or vm_config.get('disk_size_gb', 500)
    disk_type = vm_config.get('disk_type', 'pd-balanced')
    boot_image = vm_config.get('boot_image', 'projects/ml-images/global/images/c0-deeplearning-common-cu124-v20241224-debian-11-py310')
    gpu_count = vm_config.get('gpu_count', 1)
    use_spot = args.spot or vm_config.get('use_spot', False)
    
    # Get provisioning config values
    provisioning_config = config_data.get('provisioning', {})
    if args.max_zones is None:
        args.max_zones = provisioning_config.get('max_zones')
    
    # Validate required arguments
    if not project_id:
        parser.error("--project is required (either as argument or in config file)")
    
    # Validate delete-vm requires zone
    if args.delete_vm and not args.zone:
        parser.error("--delete-vm requires --zone")
    
    # Create VM configuration
    vm_config_obj = VMConfig(
        machine_type=machine_type,
        disk_size_gb=disk_size_gb,
        disk_type=disk_type,
        boot_image=boot_image,
        gpu_count=gpu_count,
        use_spot=use_spot
    )
    
    # Initialize provisioner
    provisioner = GCPGPUProvisioner(project_id, vm_config_obj)
    
    # Handle authentication action
    if args.authenticate:
        if provisioner.authenticate():
            print("\n✓ Authentication successful")
            sys.exit(0)
        else:
            print("\n✗ Authentication failed")
            sys.exit(1)
    
    # Check authentication for other actions
    if not provisioner.check_authentication():
        print("\n✗ Not authenticated. Run with --authenticate first or use 'gcloud auth login'")
        sys.exit(1)
    
    # Execute requested action
    try:
        if args.list_gpu_types:
            df = provisioner.list_all_gpu_types()
            if not df.empty:
                print("\n" + "="*80)
                print("Available GPU Types by Zone")
                print("="*80)
                print(df.to_string(index=False))
                
                if args.output:
                    df.to_csv(args.output, index=False)
                    print(f"\n✓ Results saved to {args.output}")
        
        elif args.list_zones:
            zones_with_gpu = provisioner.find_zones_with_gpu(gpu_type)
            df = pd.DataFrame(zones_with_gpu)
            
            if not df.empty:
                print("\n" + "="*80)
                print(f"Zones with {gpu_type}")
                print("="*80)
                print(df.to_string(index=False))
                print(f"\nTotal: {len(df)} zones")
                
                if args.output:
                    df.to_csv(args.output, index=False)
                    print(f"\n✓ Results saved to {args.output}")
            else:
                print(f"\n✗ No zones found with {gpu_type}")
        
        elif args.provision:
            zones_with_gpu = provisioner.find_zones_with_gpu(gpu_type)
            
            if not zones_with_gpu:
                print(f"\n✗ No zones found with {gpu_type}")
                sys.exit(1)
            
            # Limit zones if requested
            if args.max_zones:
                zones_with_gpu = zones_with_gpu[:args.max_zones]
                print(f"\nLimiting to {args.max_zones} zones")
            
            print(f"\nAttempting to provision VMs in {len(zones_with_gpu)} zones...")
            results_df = provisioner.provision_vms_batch(zones_with_gpu)
            
            print("\n" + "="*80)
            print("Provisioning Results")
            print("="*80)
            print(results_df.to_string(index=False))
            
            # Summary
            success_count = results_df[results_df['GPU Allocated'] == 'Yes'].shape[0]
            print(f"\n✓ Successfully provisioned: {success_count}/{len(zones_with_gpu)} VMs")
            
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\n✓ Results saved to {args.output}")
        
        elif args.delete_vm:
            success = provisioner.delete_vm(args.zone, args.delete_vm)
            sys.exit(0 if success else 1)
        
        elif args.list_vms:
            df = provisioner.list_running_vms()
            
            if not df.empty:
                print("\n" + "="*80)
                print("Running VM Instances")
                print("="*80)
                print(df.to_string(index=False))
                print(f"\nTotal: {len(df)} VMs")
                
                if args.output:
                    df.to_csv(args.output, index=False)
                    print(f"\n✓ Results saved to {args.output}")
            else:
                print("\n✗ No running VMs found")
    
    except KeyboardInterrupt:
        print("\n\n✗ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()