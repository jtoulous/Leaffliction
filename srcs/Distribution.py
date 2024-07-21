import matplotlib.pyplot as plt
import argparse
import os

def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('repo', help='images repo')

    return parser.parse_args()


def BarChart(repo):
    sub_repos = os.listdir(repo)
    data = {}

    for sub_repo in sub_repos:
        content = os.listdir(args.repo + '/' + sub_repo)
        data[sub_repo] = len(content)
        
    diseases = list(data.keys())
    counts = list(data.values())
    colors = plt.cm.tab20.colors

    fig = plt.figure(figsize=(10, 5))
    plt.bar(diseases, counts, color=colors[:len(diseases)], width=0.4)
    plt.xlabel('Diseases')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def PieChart(repo):
    sub_repos = os.listdir(repo)
    data = {}

    for sub_repo in sub_repos:
        content = os.listdir(args.repo + '/' + sub_repo)
        data[sub_repo] = len(content)

    diseases = list(data.keys())
    counts = list(data.values())
    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pie(counts, labels=diseases, colors=colors[:len(diseases)], autopct='%1.1f%%', startangle=140)
    ax.axis('equal') 
    plt.title('Distribution des Images par Maladie')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        args = ParseArguments()
        data = {}

        chart_type = input('Chart types:\n 1 ==> Bar chart\n 2 ==> Pie chart\n\n===> ').strip()
        if chart_type == '1':
            BarChart(args.repo)
        elif chart_type == '2':
            PieChart(args.repo)


    except Exception as error:
        print(error)