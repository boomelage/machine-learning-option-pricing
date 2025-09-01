from graphviz import Digraph

def make_math(htmlstr):
    return f'<<font face="Times" point-size="20"><i>{htmlstr}</i></font>>'

dot = Digraph()
dot.attr(rankdir='LR', ranksep='0.5', nodesep='0',dpi='600')
dot.attr('edge',arrowsize='0.3')
# dot.attr(splines='false')  # Forces straight arrows

input_layer = [
    ('01Di1', make_math('X<sub>i,1</sub>')),
    ('02elip', ' . . . '),
    ('03Dim', make_math('X<sub>i,m</sub>')),
    ('04F1Di', make_math('X<sub>i,m+1</sub>')),
    ('05elip', ' . . . '),
    ('06Fmtildi', make_math('X<sub>i,m+m&#x303;</sub>'))
]

for _,n in enumerate(input_layer):
    if n[0].find('elip')==-1:
        dot.node(n[0],n[1],width='1.2', fixedsize='true')
    else:
        dot.node(n[0],n[1],shape='plaintext')


indices = ['elip','i','elip','h',1]


hidden_layer1 = []
hidden_layer2 = []
for n, value in enumerate(indices):
    if value == 'elip':
        hidden_node1 = f'0{n}hidden1elip'
        hidden_layer1.append({'name':hidden_node1,'label':' . . . ','shape':'plaintext'})
        print(hidden_node1)
        # hidden_node2 = f'0{n}hidden2elip'
        # hidden_layer2.append({'name':hidden_node2,'label':' . . . ','shape':'plaintext'})
    else:
        hidden_node1 = f'0{n}hidden1'
        hidden_layer1.append({'name':hidden_node1,'label':make_math(f'𝛗(L<sub>input</sub> &times; w̅<sub>1,{value}</sub>)')})
        print(hidden_node1)
        # hidden_node2 = f'0{n}hidden2'
        # hidden_layer2.append({'name':hidden_node2,'label':make_math(f'𝛗(𝛗(L<sub>input</sub> &times; w̅<sub>1,{value}</sub>) &times; w̅<sub>2,{value}</sub>)')})


for i in hidden_layer1:
    print(i['name'])


for _,l in enumerate(hidden_layer1):
    dot.node(**l)

# for l in hidden_layer2:
#     dot.node(**l)

for i,inp in enumerate(input_layer):
    for j,hid1 in enumerate(hidden_layer1):
        if 'elip' in inp[0] or 'elip' in hid1['name']:
            dot.edge(inp[0],hid1['name'],style='invis')
        else:
            dot.edge(inp[0],hid1['name'])

# for i,hid1 in enumerate(hidden_layer1):
#     for j,hid2 in enumerate(hidden_layer2):
#         if 'elip' in hid1['name'] or 'elip' in hid2['name']:
#             dot.edge(hid1['name'],hid2['name'],style='invis')
#         else:
#             dot.edge(hid1['name'],hid2['name'])

dot.node('output',make_math('F(w̅<sub>1</sub>,w̅<sub>2</sub>) = 𝛗(L<sub>hidden</sub> &times; w̅<sub>2</sub>)'))

for i,hid in enumerate(hidden_layer1):
    if 'elip' in hid['name']:
        dot.edge(hid['name'],'output',style='invis')
    else:
        dot.edge(hid['name'],'output')

dot.render('elip',format='png',view=True)
