import ast, os

def transform_Dockground_name(pair, sample):
    if '\uf03a' in pair:
        pair1, pair2 = pair.split('\uf03a')
        pair2 = pair1.split('_')[0] + '_' + pair2
        sample_id = sample.split('_')[2]
        return pair1 + '--' + pair2, pair1 + '--' + pair2 + '_' + sample_id
    elif len(pair.split('_')) == 3:
        base, pair1, pair2 = pair.split('_')
        pair1 = base + '_' + pair1
        pair2 = base + '_' + pair2
        sample_id = sample.split('_')[3]
        return pair1 + '--' + pair2, pair1 + '--' + pair2 + '_' + sample_id
    else:
        raise ValueError('Dockground utils: problem with pair {}'.format(pair))

def select_Dockground_split(folder_path, dataset):
    groups_txt_path = os.path.join(folder_path, 'groups.txt')
    with open(groups_txt_path) as f:
        data = f.read()

    data = data.split('=')[1].replace(' ', '').strip().upper()
    split_dict = ast.literal_eval(data)

    splits = [[] for f in range(len(split_dict))]
    for d in dataset:
        pair1, pair2 = d.pair.split('--')
        for key_fold, values in enumerate(split_dict.values()): # Use enumerate() to assure presence of correct keys (i.e. from 0 to len(split_dict) - 1)
            for v in values:
                if v in pair1.upper() and v in pair2.upper():
                    splits[key_fold].append(d)
                elif v not in pair1.upper() and v not in pair2.upper():
                    continue
                else:
                    raise ValueError('Dockground utils: problem with pair {}'.format(d.pair))
    
    predefined_splits = [{'train':[], 'val':[]} for f in range(len(split_dict))]
    for k in range(len(split_dict)):
        predefined_splits[k]['val'] = splits[k]
        for fold in range(len(split_dict)):
            if fold != k:
                predefined_splits[k]['train'] += splits[fold]
    return predefined_splits