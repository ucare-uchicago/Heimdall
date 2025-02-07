#!/bin/bash
set -e
set -u
set -o pipefail

green="\e[32m"
red="\e[31m"
reset="\e[0m"

# ================================== 1. Check SSD Requirement =========================================
# Function to check if a device is an SSD (rotational == 0)
is_ssd() {
  local device="$1"
  local rotational=$(lsblk -no rota "/dev/$device" 2>/dev/null)
  if [[ "$rotational" =~ "0" ]]; then
    return 0  # It's an SSD
  else
    return 1  # It's not an SSD
  fi
}

# Get a list of all unmounted block devices with no mounted partitions
unmounted_devices=($(lsblk -o name,mountpoint -n | awk '$2 == "" {print $1}' | grep -E '^[a-zA-Z0-9]+$'))

unmounted_ssd_count=0
unmounted_ssd_names=""

# Check if they are SSDs
for device in "${unmounted_devices[@]}"; do
  if is_ssd "$device"; then
    # Check if there are no mounted partitions for this device
    if ! mount | grep -q "/dev/$device"; then
      unmounted_ssd_count=$((unmounted_ssd_count + 1))
      unmounted_ssd_names="$unmounted_ssd_names $device"
    fi
  fi
done

# Check if there are at least two unmounted SSDs
if ((unmounted_ssd_count >= 2)); then
  echo -e "${green}✔ There are at least two unmounted SSDs: $unmounted_ssd_names ${reset}"
else
  echo -e "${red}✘ Error: There are not enough unmounted SSDs (less than 2). ${reset}"
fi


# ================================== 2. Check Whether Ubuntu20.04 =========================================
ubuntu_version=$(lsb_release -r -s)

# Check if the version is 20.04
if [ "$ubuntu_version" == "20.04" ]; then
  echo -e "${green}✔ OS Version: Ubuntu 20.04.${reset}"
else
  echo -e "${red}✘ Error: We recommand to use Ubuntu 20.04 to conduct our experiment.${reset}"
  exit 1
fi

# ================================== 3. Install Dependencies =========================================
sudo apt-get update
sudo apt-get -y install build-essential tmux git pkg-config cmake zsh
sudo apt-get -y install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libiberty-dev autoconf zstd
sudo apt-get -y install libreadline-dev binutils-dev libnl-3-dev
sudo apt-get -y install ecryptfs-utils cpufrequtils
sudo apt-get install dwarves
sudo apt-get install nvidia-cuda-toolkit
pip3 install --upgrade pip
pip3 install numpy matplotlib scipy
pip3 show numpy
pip3 install tensorflow
pip3 install keras
pip3 install pandas
pip3 install scikit-learn
pip3 install statsmodels


link_path="/usr/include/numpy"
user_site_packages=$(python3 -m site --user-site)

# Check if the symbolic link or directory already exists
if [ ! -e "$link_path" ]; then
  # Create a symbolic link for numpy include files
  sudo ln -s "$user_site_packages/numpy/core/include/numpy/" "$link_path"
else
  echo "Symbolic link '$link_path' already exists. Skipping."
fi


echo -e "${green}✔ Dependencies Installed.${reset}"