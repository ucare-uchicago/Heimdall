#!/bin/bash
set -e
set -u
set -o pipefail

green="\e[32m"
red="\e[31m"
reset="\e[0m"


if [ "$(id -u)" -ne 0 ]; then
    echo -e "${red}✘ Error: This script will modify /etc/default/grub! Must be run as root. ${reset}"
    exit 1
fi


# ================================== Kernel Compilation (15~25 mins) =========================================
cd ../heimdall-linux-6.0
./full_compilation.sh
echo -e "${green}✔ Kernel compilation done ${reset}"


# ================================== GRUB Configuration Update =========================================
GRUB_DEFAULT="gnulinux-advanced-LABEL=cloudimg-rootfs>gnulinux-6.0.0-heimdall-advanced-LABEL=cloudimg-rootfs"
GRUB_CMDLINE_LINUX="quiet splash cma=128M@0-4G log_buf_len=16M nvme_core.multipath=0"
GRUB_CMDLINE_LINUX_DEFAULT="console=tty0 console=ttyS0,115200 no_timer_check nofb nomodeset gfxpayload=text cma=128M@0-4G log_buf_len=16M"

# Copy the original /etc/default/grub
cp /etc/default/grub /etc/default/grub.backup
echo -e "${green}✔ The original /etc/default/grub is copied to /etc/default/grub.backup ${reset}"

# Change the grub configuration
## Change GRUB_CMDLINE_LINUX_DEFAULT. There are two GRUB_CMDLINE_LINUX_DEFAULT in grub, and we target to change the last one.
sudo tac /etc/default/grub | \
sed "0,/GRUB_CMDLINE_LINUX_DEFAULT=.*/{//s//GRUB_CMDLINE_LINUX_DEFAULT=\"$GRUB_CMDLINE_LINUX_DEFAULT\"/}" | \
tac > /tmp/grub_modified

mv /tmp/grub_modified /etc/default/grub

## Change GRUB_DEFAULT
sed -i "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"$GRUB_DEFAULT\"|" /etc/default/grub
## Change GRUB_CMDLINE_LINUX
sed -i "s|^GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX=\"$GRUB_CMDLINE_LINUX\"|" /etc/default/grub

update-grub

echo -e "${green}✔ GRUB configuration has been updated successfully.${reset}"