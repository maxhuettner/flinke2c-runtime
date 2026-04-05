#!/usr/bin/env bash
set -euo pipefail

username="${REMOTE_USER:-dev}"
password="${SSH_PASSWORD:-dev}"
ssh_public_key="${SSH_PUBLIC_KEY:-}"
ssh_public_key_file="${SSH_PUBLIC_KEY_FILE:-}"

if id "${username}" >/dev/null 2>&1; then
    echo "${username}:${password}" | chpasswd

    home_dir="$(getent passwd "${username}" | cut -d: -f6)"
    ssh_dir="${home_dir}/.ssh"
    auth_keys="${ssh_dir}/authorized_keys"

    install -d -m 700 -o "${username}" -g "${username}" "${ssh_dir}"

    if [[ -n "${ssh_public_key_file}" && -f "${ssh_public_key_file}" ]]; then
        cp "${ssh_public_key_file}" "${auth_keys}"
        chown "${username}:${username}" "${auth_keys}"
        chmod 600 "${auth_keys}"
    elif [[ -n "${ssh_public_key}" ]]; then
        printf '%s\n' "${ssh_public_key}" > "${auth_keys}"
        chown "${username}:${username}" "${auth_keys}"
        chmod 600 "${auth_keys}"
    fi
fi

exec "$@"
