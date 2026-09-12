#!/bin/sh
# Generates s3.json from environment variables so credentials live in .env

set -eu

mkdir -p /etc/seaweedfs

cat > /etc/seaweedfs/s3.json <<JSON
{
  "identities": [
    {
      "name": "app",
      "credentials": [
        { "accessKey": "${S3_ACCESS_KEY}", "secretKey": "${S3_SECRET_KEY}" }
      ],
      "actions": ["Admin", "Read", "Write", "List", "Tagging"]
    }
  ]
}
JSON

/bucket-init.sh

exec weed "$@"
