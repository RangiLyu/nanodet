import multiprocessing as mp
import os
import platform

import cv2

from nanodet.util.env_utils import set_multi_processing


def test_setup_multi_processes():
    # temp save system setting
    sys_start_mehod = mp.get_start_method(allow_none=True)
    sys_cv_threads = cv2.getNumThreads()
    # pop and temp save system env vars
    sys_omp_threads = os.environ.pop("OMP_NUM_THREADS", default=None)
    sys_mkl_threads = os.environ.pop("MKL_NUM_THREADS", default=None)

    # test distributed
    set_multi_processing(distributed=True)
    assert os.getenv("OMP_NUM_THREADS") == "1"
    assert os.getenv("MKL_NUM_THREADS") == "1"
    # when set to 0, the num threads will be 1
    assert cv2.getNumThreads() == 1
    if platform.system() != "Windows":
        assert mp.get_start_method() == "fork"

    # test num workers <= 1
    os.environ.pop("OMP_NUM_THREADS")
    os.environ.pop("MKL_NUM_THREADS")
    set_multi_processing(distributed=False)
    assert "OMP_NUM_THREADS" not in os.environ
    assert "MKL_NUM_THREADS" not in os.environ

    # test manually set env var
    os.environ["OMP_NUM_THREADS"] = "4"
    set_multi_processing(distributed=False)
    assert os.getenv("OMP_NUM_THREADS") == "4"

    # test manually set opencv threads and mp start method
    config = dict(mp_start_method="spawn", opencv_num_threads=4, distributed=True)
    set_multi_processing(**config)
    assert cv2.getNumThreads() == 4
    assert mp.get_start_method() == "spawn"

    # revert setting to avoid affecting other programs
    if sys_start_mehod:
        mp.set_start_method(sys_start_mehod, force=True)
    cv2.setNumThreads(sys_cv_threads)
    if sys_omp_threads:
        os.environ["OMP_NUM_THREADS"] = sys_omp_threads
    else:
        os.environ.pop("OMP_NUM_THREADS")
    if sys_mkl_threads:
        os.environ["MKL_NUM_THREADS"] = sys_mkl_threads
    else:
        os.environ.pop("MKL_NUM_THREADS")
