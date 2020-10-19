package com.rangi.nanodet;

import androidx.annotation.NonNull;

class AppCrashHandler implements Thread.UncaughtExceptionHandler {

    private Thread.UncaughtExceptionHandler uncaughtExceptionHandler = Thread.getDefaultUncaughtExceptionHandler();

    @Override
    public void uncaughtException(@NonNull Thread t, @NonNull Throwable e) {
        uncaughtExceptionHandler.uncaughtException(t, e);
    }

    public static void register() {
        Thread.setDefaultUncaughtExceptionHandler(new AppCrashHandler());
    }

}
