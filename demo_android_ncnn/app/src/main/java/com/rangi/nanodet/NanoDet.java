package com.rangi.nanodet;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class NanoDet {
    static {
        System.loadLibrary("yolov5");
    }

    public static native void init(AssetManager manager, boolean useGPU);
    public static native Box[] detect(Bitmap bitmap, double threshold, double nms_threshold);
}
