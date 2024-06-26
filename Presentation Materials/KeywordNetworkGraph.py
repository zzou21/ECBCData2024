# Attempting to use NetworkX to showcase keyword relations. It does not work well. Do not use before the "In-progress" below is deleted:

# In-progress

# Author: Jerry Zou

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#keywords used for testing purposes
categories = {
    "Patrick Copland": ["Danger", "Deliverance", "Distresse", "fruitfull Countrie", "Corne", "Profit and Pleasure", "Tribe", "Heathen", "Strangers", "Opachancano", "Opachankano", "mollifie", "Preacher", "Instructed", "Gospel", "Sathan", "Iohn Rolfe", "College Land", "Baptized", "Epistles", "Posteritie", "Purse", "Scarce", "Your people", "Labour", "Perpetuall warre", "Natural inhabitant", "Stranger and alien", "barbarian", "conversant", "Epicureanism", "Millenarianism", "Worldings", "Mortal", "Prophet", "Thankesgiuing"
    ],
    "John Brinsley": ["Barbarous", "Ciuilite", "Ministerie", "magistracie", "Learning", "Growth", "Labour", "schoole-labours", "Soules", "poore soules", "Sauage", "Indian", "Ignornace", "blindnesse", "Idolatrie", "Vngodlie", "Instruction", "Saluation", "Charitablie", "Sathan", "Ruder countries", "Submit", "submit themselues", "Commodities", "Possession", "Natiues", "Shining light", "Light", "Salomon", "Blindness", "Seminaries", "Arts", "Roman Antichrist"
    ]
}

G = nx.Graph()
keyword_categories = {}
for category, keywords in categories.items():
    G.add_node(category, size=1000)
    for keyword in keywords:
        G.add_node(keyword, size=500)
        G.add_edge(category, keyword)
        if keyword not in keyword_categories:
            keyword_categories[keyword] = []
        keyword_categories[keyword].append(category)

pos = {
    "Patrick Copland": (-1.5, 0),
    "John Brinsley": (1.5, 0)
}
def circular_positions(center, num_nodes, radius):
    angle_step = 2 * np.pi / num_nodes
    return {
        f"node_{i}": (
            center[0] + radius * np.cos(i * angle_step),
            center[1] + radius * np.sin(i * angle_step)
        )
        for i in range(num_nodes)
    }
radius = 1  # Distance from category to keywords
for category, keywords in categories.items():
    category_pos = pos[category]
    keyword_positions = circular_positions(category_pos, len(keywords), radius)
    for i, keyword in enumerate(keywords):
        if keyword not in pos:
            pos[keyword] = keyword_positions[f"node_{i}"]
            
vertical_offset = 0.5 
shared_keyword_y = {}

for keyword, categories in keyword_categories.items():
    if len(categories) > 1:
        avg_x = np.mean([pos[cat][0] for cat in categories])
        avg_y = np.mean([pos[cat][1] for cat in categories])
        if avg_x not in shared_keyword_y:
            shared_keyword_y[avg_x] = avg_y
        else:
            shared_keyword_y[avg_x] += vertical_offset
        pos[keyword] = (avg_x, shared_keyword_y[avg_x])

sizes = [nx.get_node_attributes(G, 'size').get(node, 100) for node in G.nodes()]
nx.draw(G, pos, with_labels=True, node_size=sizes, font_size=8, node_color='skyblue', edge_color='gray')
plt.title("Keyword Network Graph")
plt.show()