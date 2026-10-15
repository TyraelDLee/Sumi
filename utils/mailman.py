import subprocess

# def mailman(exp_name, log_file):
#     # with open(log_file, 'r') as f:
#     #     content = f.read()
#     with open(log_file, 'r') as f:
#         content = [line.strip() for line in f if line.strip()]
#     # print("\n".join(content))
#     results = "\n".join(content)
#     subprocess.run(["./utils/mailman.sh", exp_name, " "])
import sys, os, time
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mailman.sh")))
def mailman(exp_name, content_src):

    with open(content_src, 'r') as f:
        original_content = f.read()

    # Write the new lines followed by the original content
    with open(content_src, 'w') as f:
        f.write("From : diffpmaeproject@gmail.com\nTo: lee.tyreal12138@outlook.com\nSubject: Experiment {exp_name} is finished\n\n".format(exp_name = exp_name) + original_content)

    subprocess.run([os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mailman.sh")), exp_name, content_src])

    with open(content_src, 'w') as f:
        f.write(original_content)


