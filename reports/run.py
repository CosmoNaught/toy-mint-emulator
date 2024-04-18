from orderly.run import orderly_run

path = "X:\\Cosmo\\projects\\toy-mint-emulator\\src"

id0 = orderly_run("data", root=path)
id1 = orderly_run("benchmark", root=path)
id2 = orderly_run("post", root=path)
#python -u "x:\Cosmo\projects\toy-mint-emulator\reports\run.py"