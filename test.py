import networkx as nx, json
import matplotlib.pyplot as plt

# Initialize the data
data = {
    'earning': {'money': ['A16864', 'A19313'], 'christ': ['A16864'], 'darkenesse': ['A16864'], 'naked': ['A16864'], 'wild': ['A16864'], 'light': ['A19313']},
    'religion': {'christ': ['A16864', 'A19313', 'A14803'], 'force': ['A16864', 'A73849', 'A19313', 'A14803'], 'light': ['A73849', 'A19313', 'A14803'], 'voluntary': ['A73849'], 'money': ['A19313']},
    'gouern∣ment': {'darkenesse': ['A16864'], 'clothed': ['A16864'], 'naked': ['A16864'], 'cultivated': ['A16864'], 'wild': ['A16864'], 'voluntary': ['A16864'], 'force': ['A16864']},
    'barbarous': {'naked': ['A16864', 'A14803'], 'darkenesse': ['A14803'], 'wild': ['A14803'], 'voluntary': ['A14803'], 'force': ['A14803']},
    'to': {'cultivated': ['A16864', 'A73849', 'A19313', 'A14803'], 'clothed': ['A73849', 'A19313'], 'wild': ['A73849'], 'voluntary': ['A73849', 'A19313'], 'naked': ['A19313']},
    'vnderminable': {'money': ['A73849'], 'darkenesse': ['A73849'], 'naked': ['A73849'], 'voluntary': ['A73849'], 'force': ['A73849']},
    'people': {'money': ['A73849', 'A19313', 'A14803'], 'cultivated': ['A73849'], 'voluntary': ['A73849', 'A19313'], 'force': ['A73849', 'A19313'], 'christ': ['A19313'], 'light': ['A19313', 'A14803'], 'clothed': ['A14803']},
    'papers': {'money': ['A73849'], 'darkenesse': ['A73849'], 'clothed': ['A73849'], 'naked': ['A73849'], 'cultivated': ['A73849'], 'wild': ['A73849'], 'voluntary': ['A73849'], 'force': ['A73849']},
    'moneyes': {'money': ['A73849'], 'darkenesse': ['A73849'], 'clothed': ['A73849'], 'naked': ['A73849'], 'cultivated': ['A73849'], 'voluntary': ['A73849']},
    'leuied': {'money': ['A73849'], 'darkenesse': ['A73849'], 'clothed': ['A73849'], 'naked': ['A73849'], 'cultivated': ['A73849'], 'voluntary': ['A73849']},
    'determines': {'money': ['A73849'], 'darkenesse': ['A73849'], 'clothed': ['A73849'], 'naked': ['A73849'], 'cultivated': ['A73849'], 'wild': ['A73849'], 'voluntary': ['A73849'], 'force': ['A73849']},
    'benefit': {'money': ['A73849'], 'cultivated': ['A73849', 'A14803'], 'force': ['A73849', 'A19313', 'A14803'], 'darkenesse': ['A14803'], 'clothed': ['A14803'], 'voluntary': ['A14803']},
    'samaria': {'christ': ['A73849'], 'light': ['A73849'], 'darkenesse': ['A73849'], 'voluntary': ['A73849'], 'force': ['A73849']},
    'papists': {'christ': ['A73849'], 'darkenesse': ['A73849'], 'clothed': ['A73849'], 'naked': ['A73849'], 'voluntary': ['A73849']},
    'plantation': {'light': ['A73849', 'A19313', 'A14803'], 'voluntary': ['A73849', 'A19313'], 'force': ['A73849'], 'money': ['A19313', 'A14803'], 'christ': ['A19313', 'A14803']},
    'virginia': {'darkenesse': ['A73849', 'A19313'], 'clothed': ['A73849', 'A19313', 'A14803'], 'naked': ['A73849', 'A19313'], 'force': ['A73849', 'A19313'], 'christ': ['A14803']},
    'vehiculum': {'darkenesse': ['A73849'], 'clothed': ['A73849'], 'naked': ['A73849'], 'cultivated': ['A73849'], 'voluntary': ['A73849']},
    'proceedings': {'darkenesse': ['A73849'], 'clothed': ['A73849'], 'naked': ['A73849'], 'cultivated': ['A73849'], 'voluntary': ['A73849'], 'force': ['A73849']},
    'aduancement': {'darkenesse': ['A73849'], 'naked': ['A73849'], 'voluntary': ['A73849', 'A14803'], 'money': ['A14803'], 'light': ['A14803']},
    'vnconuerted': {'money': ['A19313'], 'christ': ['A19313'], 'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'wild': ['A19313'], 'voluntary': ['A19313']},
    'vir∣ginia': {'money': ['A19313'], 'christ': ['A19313'], 'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'onesymus': {'money': ['A19313'], 'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'ofitable': {'money': ['A19313'], 'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'misery': {'money': ['A19313'], 'christ': ['A19313'], 'darkenesse': ['A14803'], 'naked': ['A14803'], 'voluntary': ['A14803']},
    'miscarry': {'money': ['A19313'], 'darkenesse': ['A19313'], 'naked': ['A19313'], 'voluntary': ['A19313'], 'force': ['A19313']},
    'go∣uernment': {'money': ['A19313'], 'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'burden': {'money': ['A19313'], 'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'transporting': {'christ': ['A19313'], 'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'worshipfull': {'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'vnp': {'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'philem': {'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'ho∣nourable': {'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'fashion': {'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'voluntary': ['A19313']},
    'consecrating': {'darkenesse': ['A19313'], 'clothed': ['A19313'], 'naked': ['A19313'], 'cultivated': ['A19313'], 'wild': ['A19313'], 'voluntary': ['A19313'], 'force': ['A19313']},
    'surfetted': {'christ': ['A14803'], 'darkenesse': ['A14803'], 'clothed': ['A14803'], 'naked': ['A14803'], 'cultivated': ['A14803']},
    'shining': {'christ': ['A14803'], 'darkenesse': ['A14803'], 'clothed': ['A14803'], 'naked': ['A14803'], 'voluntary': ['A14803']},
    'mas∣sacre': {'christ': ['A14803'], 'darkenesse': ['A14803'], 'naked': ['A14803'], 'voluntary': ['A14803'], 'force': ['A14803']},
    'en∣glish': {'light': ['A14803'], 'darkenesse': ['A14803'], 'naked': ['A14803'], 'voluntary': ['A14803'], 'force': ['A14803']},
    'sup∣posing': {'darkenesse': ['A14803'], 'clothed': ['A14803'], 'naked': ['A14803'], 'voluntary': ['A14803'], 'force': ['A14803']},
    'endeuored': {'darkenesse': ['A14803'], 'clothed': ['A14803'], 'naked': ['A14803'], 'cultivated': ['A14803'], 'voluntary': ['A14803']}
}

# with open("/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/networkJSON.json", "w") as file:
#     json.dump(data, file, indent=4)
wordSet = set()
categorySet = set()
yearSet = set()

with open("/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/networkJSON.json", "r") as file:
    content = json.load(file)
    # print(content)
    for word, dict in content.items():
        for category, list in dict.items():
            # print(word, category)
            for file in list:
                wordSet.add(word)
                categorySet.add(category)
                yearSet.add(file)
print(wordSet)
print(categorySet)
print(yearSet)
for word in categorySet:
    print(word)

                


# Initialize the graph
# G = nx.Graph()

# # Add nodes and edges
# for word, categories in data.items():
#     G.add_node(word, type='word')
#     for category, items in categories.items():
#         G.add_node(category, type='category')
#         G.add_edge(word, category)
#         for item in items:
#             G.add_node(item, type='item')
#             G.add_edge(category, item)

# # Draw the graph with different colors for each type of node
# pos = nx.spring_layout(G, seed=42)  # positions for all nodes
# plt.figure(figsize=(12, 8))

# # Draw nodes
# word_nodes = [node for node, attrs in G.nodes(data=True) if attrs['type'] == 'word']
# category_nodes = [node for node, attrs in G.nodes(data=True) if attrs['type'] == 'category']
# item_nodes = [node for node, attrs in G.nodes(data=True) if attrs['type'] == 'item']

# nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_size=500, node_color='lightblue', label='Words')
# nx.draw_networkx_nodes(G, pos, nodelist=category_nodes, node_size=500, node_color='lightgreen', label='Categories')
# nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_size=500, node_color='lightcoral', label='Items')

# # Draw edges
# nx.draw_networkx_edges(G, pos, width=2, alpha=0.6)

# # Draw labels
# nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

# # Add legend
# plt.legend(scatterpoints=1)
# plt.title('Network Graph of Words, Categories, and Their Associations')
# plt.show()