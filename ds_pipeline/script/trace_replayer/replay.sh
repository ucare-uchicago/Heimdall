#!/bin/bash

while [ $# -gt 0 ]; do
  case "$1" in
    -user)
      user="$2" # to regain the access from the sudo user
      ;;
    -device)
      device="$2"
      ;;
    -dir)
      dir="$2"
      ;;
    -file)
      file="$2"
      ;;
    -pattern)
      pattern="$2"
      ;;
    -output_dir)
      output_dir="$2"
      ;;
    *)
      printf "ERROR: Invalid argument. \n(sample: ./replay.sh -device /dev/nvme0n1 -dir \$FLASHNET/data/trace_raw/ -pattern "*cut*trace" -output_dir \$FLASHNET/data/trace_profile/)\n"
      exit 1
  esac
  shift
  shift
done

dev_name=${device//"/dev/"/} # remove the "/dev/" -> we just need the device name
# echo "user: ${user}"

function generate_output_path()
{
    filename=$(basename ${file})
    echo "$output_dir/$dev_name/$filename"
}

function generate_stats_path()
{
    filename=$(basename ${file})
    echo "$output_dir/$dev_name/$filename.stats"
}

function replay_file()
{
    echo ""
    echo "Replaying on ${dev_name} : ${file}"
    output_path=$(generate_output_path)
    stats_path=$(generate_stats_path)
    sudo ./io_replayer $device $file $output_path 
    echo "output replayed trace : ${output_path}"
    echo "         output stats : ${stats_path}"
    chown -R $user "$output_dir/$dev_name"
    chown $user "$output_path"
    chown $user "$stats_path"
}

if [[ $device && $dir && $pattern && $output_dir ]]; then
    # Iterate through the files in dir 
    for file in ${dir}/*; do
        if [[ -f $file ]]; then # check this file
            # check whether it satisfy the pattern
            if ../pattern_checker.py -pattern ${pattern} -file ${file} | grep -q 'True'; then
                replay_file
            fi
        fi
    done
elif [[ $device && $file && $output_dir ]]; then
    replay_file 
fi
