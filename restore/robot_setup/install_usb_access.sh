#!/usr/bin/env bash
set -euo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sudo install -m 0644 "$setup_dir/99-so101.rules" /etc/udev/rules.d/99-so101.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
printf '%s\n' 'USB rule installed. Reconnect the controller if access is not refreshed.'
