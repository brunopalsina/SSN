import os

folder = "/home/bruno/Uni/SSN/SIESTA_examples-main/MD_reaction/practice_session/imagesequence"
print(os.listdir(folder))
for filename in os.listdir(folder):
    old_name = os.path.join(folder, filename)
    base = os.path.splitext(old_name)[0]
    os.rename(old_name, base + ".png")