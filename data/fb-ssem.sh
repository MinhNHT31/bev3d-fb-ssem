#!/bin/bash

TARGET_DIR="FB-SSEM"
mkdir -p "$TARGET_DIR"

# URL list
URLS="
https://fb-ssem.s3.us-west-2.amazonaws.com/CameraCalibrationParameters/camera_intrinsics.yml
https://fb-ssem.s3.us-west-2.amazonaws.com/CameraCalibrationParameters/camera_positions_for_extrinsics.txt
https://fb-ssem.s3.us-west-2.amazonaws.com/LICENSE.txt
https://fb-ssem.s3.us-west-2.amazonaws.com/README.pdf
https://fb-ssem.s3.us-west-2.amazonaws.com/images0.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images1.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images2.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images3.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images4.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images5.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images6.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images7.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images8.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images9.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images10.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images11.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images12.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images13.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images14.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images15.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images16.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images17.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images18.zip
https://fb-ssem.s3.us-west-2.amazonaws.com/images19.zip
"

echo "$URLS" > url_list.txt

echo "🔽 Downloading..."
aria2c -j 10 -x 16 -s 16 -i url_list.txt

echo "📦 Extracting ZIP files..."
for f in images*.zip; do
    unzip -q "$f" -d "$TARGET_DIR"
done

# Move other files
mkdir -p "$TARGET_DIR/CameraCalibrationParameters"
mv camera_intrinsics.yml "$TARGET_DIR/CameraCalibrationParameters" 2>/dev/null
mv camera_positions_for_extrinsics.txt "$TARGET_DIR/CameraCalibrationParameters" 2>/dev/null
mv LICENSE.txt "$TARGET_DIR" 2>/dev/null
mv README.pdf "$TARGET_DIR" 2>/dev/null

echo "🎉 Done!"