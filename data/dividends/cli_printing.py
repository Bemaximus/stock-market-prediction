# Print iterations progress
# Adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console#answer-34325723
def print_progress_bar(iteration, total, prefix = '', suffix = '', 
	decimals = 1, length = 100, fill = '█', fraction = False, printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        fraction    - Optional  : display fraction complete (Bool)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if fraction:
    	fraction = f" {iteration}/{total}"
    else:
    	fraction = ""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {fraction} {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()