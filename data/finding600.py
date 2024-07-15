import os

pathName = "/Users/lucasma/Downloads/AllVirginia"
with open ("/Users/lucasma/Downloads/600.txt", "a") as f:
    for filename in os.listdir(pathName):
        f.write(f"{filename}\n")
