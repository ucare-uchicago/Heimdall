#!/bin/bash

while [ $# -gt 0 ]; do
  case "$1" in
    -user)
      user="$2" # to regain the access from the sudo user
      ;;
    -original_device_index)
      original_device_index="$2"
      ;;
    -devices_list)
      devices_list="$2"
      ;;
    -trace)
      trace="$2"
      ;;
    -output_dir)
      output_dir="$2"
      ;;
    -duration)
      duration="$2"
      ;;
    *)
      printf "ERROR: Invalid argument. Check the readme file.\n"
      exit 1
  esac
  shift
  shift
done
output_file="${output_dir}/$(basename ${trace})"
trace_dir=$(dirname ${trace})
echo "user: ${user}"
echo "original_device_index: ${original_device_index}"
echo "trace_dir: ${trace_dir}"
echo "trace: ${trace}"
echo "devices_list: ${devices_list}"
echo "output_dir: ${output_dir}"
echo "output_file: ${output_file}"
echo "duration: ${duration}"

mkdir -p "$output_dir/"

function generate_stats_path()
{
    trace_name=$(basename ${trace})
    echo "$output_dir/$trace_name.stats"
}

function generate_cpu_overhead_path()
{
    trace_name=$(basename ${trace})
    echo "$output_dir/$trace_name.csv"
}

function replay_file()
{
    # echo ""
    # sleep 2
    echo "prepare to run Linnos Client"
    stats_path=$(generate_stats_path)
    cpu_overhead_path=$(generate_cpu_overhead_path)
    grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> $cpu_overhead_path
    sudo ./io_replayer $original_device_index $devices_list $trace $output_file $duration
    grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> $cpu_overhead_path
    echo "output replayed trace : ${output_file}"
    echo "         output stats : ${stats_path}"
    chown -R $user:$user "$trace_dir" # needed when running as root; to remove the sudo ownership
}

if [[ $devices_list && $trace && $output_file ]]; then
    replay_file 
else
    echo "ERROR: Invalid arguments. Check the readme file."
    exit 1
fi