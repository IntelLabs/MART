import numpy as np

x = np.random.rand(1, 960, 1280, 3).astype(np.float32)

y = [
    {
        "area": np.array(
            [
                154,
                286,
                226,
            ]
        ),
        "boxes": np.array(
            [
                [1238.0, 59.0, 1259.0, 85.0],
                [739.0, 405.0, 762.0, 438.0],
                [838.0, 361.0, 853.0, 393.0],
            ],
            dtype=np.float32,
        ),
        "id": np.array(
            [
                80,
                81,
                82,
            ]
        ),
        "image_id": np.array(
            [
                16681727,
                16681727,
                16681727,
            ]
        ),
        "is_crowd": np.array(
            [
                False,
                False,
                False,
            ]
        ),
        "labels": np.array([1, 1, 1]),
    }
]

y_patch_metadata = [
    {
        "avg_patch_depth": np.array(25.20819092),
        "gs_coords": np.array([[969, 64], [1033, 92], [469, 214], [439, 166]], dtype=np.int32),
        "mask": np.zeros((960, 1280, 3), dtype=np.uint8),
        "max_depth_perturb_meters": np.array(3.0),
    }
]

batch = {"x": x, "y": y, "y_patch_metadata": y_patch_metadata}
