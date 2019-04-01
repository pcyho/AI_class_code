import json

with open('data.json', 'r') as f:
    Tree = json.load(f)
    print(Tree.get('color'))


def find(data, searchTree, best_label):
    """
    data:
    searchTree:
    best_label:
    """
    data[list(searchTree)[0]]


data = {'color': 'blake',
        'root': 'curl_up',
        'koncks': 'heavily',
        'texture': 'blur',
        'navel': 'even',
        'touch': 'hard_smooth',
        'density': 0.44,
        }

find(data, Tree)
