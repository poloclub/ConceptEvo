def parse_bool_arg(arg):
    """Parse boolean argument
    
    Args:
        - arg: argument

    Returns:
        - bool_arg: booleanized arg, either True or False

    Raises:
        - ArgumentTypeError when an invalid input is given.
    """

    true_vals = ['yes', 'true', 't', 'y', '1']
    false_vals = ['no', 'false', 'f', 'n', '0']
    valid_true_vals = gen_valid_bool_options(true_vals)
    valid_false_vals = gen_valid_bool_options(false_vals)
    
    if isinstance(arg, bool):
        bool_arg = arg
    if arg in valid_true_vals:
        bool_arg = True
    elif arg in valid_false_vals:
        bool_arg = False
    else:
        log = f'Boolean value expected for {arg}. '
        log += f'Available options for {arg}=True: {valid_true_vals}. '
        log += f'Available options for {arg}=False: {valid_false_vals}. '
        raise argparse.ArgumentTypeError(log)
    return bool_arg


def gen_valid_bool_options(opts):
    """Generate a list of the whole valid options for a boolean argument.

    A boolean argument accepts 
    - 'true', 'True', or 'TRUE' for True value
    - 'false', 'False', or 'FALSE' for False value
    
    Args: 
        - opts: a list of options for an argument.

    Returns:
        - all_valid_opts: a list of all valid options for given opts.

    Example:
        >>> opts = ['true', 'FALSE']
        >>> all_valid_opts = gen_valid_bool_options(opts)
        >>> all_valid_opts
        ['true', 'True', 'TRUE', 'false', 'False', 'FALSE']
    """

    all_valid_opts = []
    for opt in opts:
        opt_lower = opt.lower()
        opt_upper = opt.upper()
        opt_lower_with_fst_upper = opt_upper[0] + opt_lower[1:]
        all_valid_opts += [
            opt_lower, opt_upper, opt_lower_with_fst_upper
        ]
    return all_valid_opts