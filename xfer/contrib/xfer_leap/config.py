DEFAULT_CONFIG_SYNTHETIC = {
    # Number examples per task and input dimensionality
    "num_examples_per_task": 1000,
    "dim": 2,
    # task dependent bias and task independent bias
    "global_bias": 4.0,
    "task_bias": True,
    # Standard deviation of the covariates and the additive noise to the output
    "std_x": 1.0,
    "std_noise": 0.01,
    # Total number of tasks used for training/test
    "num_tasks_train": 3,
    "num_tasks_test": 10,
    "num_tasks_val": 2,
    # Total number of examples in each task that are hold out
    "hold_out": 100,
    }


DEFAULT_CONFIG_OMNIGLOT = {
    # Total number of tasks used for training/test
    "num_tasks_train": 3,
    "num_tasks_test": 10,
    # Total number of tasks used for validation
    # if None, it uses all the ones not used for training/test
    "num_tasks_val": None,
    # Total number of examples in each task that are hold out
    "hold_out": 5,
    # Transformations applied before loading the minibatch
    # ----------------------------------------------------
    # Transform the PIL image (before loading it in an mx.ndarray)
    "transform_image": None,
    # Transform the mx.ndarray before
    "transform_mxnet": None
    }