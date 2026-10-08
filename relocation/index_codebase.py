"""Incrementally index relocation into existing Graphify artifacts using Python AST.

No external Graphify executable is installed in this workspace. This records exact
files/symbols/imports and explicit integration edges; it does not invent semantic
analysis or change Graphify's previous built_at_commit provenance.
"""
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from .config import REPO
from .adapters import repo_revision


def extend_html(path, graph, community):
    """Preserve Graphify's viewer and append the same reviewed AST supplement."""
    if not path.exists():
        return
    html = path.read_text(encoding='utf-8')

    def array(name, replacement=None):
        nonlocal html
        start = html.index('const ' + name + ' = ') + len('const ' + name + ' = ')
        value, length = json.JSONDecoder().raw_decode(html[start:])
        if replacement is not None:
            html = html[:start] + json.dumps(replacement, ensure_ascii=False) + html[start+length:]
        return value

    nodes = [n for n in array('RAW_NODES') if not n['id'].startswith('relocation-index:')]
    edges = [e for e in array('RAW_EDGES') if not str(e['from']).startswith('relocation-index:') and not str(e['to']).startswith('relocation-index:')]
    supplement = [n for n in graph['nodes'] if n['id'].startswith('relocation-index:')]
    degrees = {}
    for edge in graph['links']:
        for node_id in [edge['source'], edge['target']]:
            degrees[node_id] = degrees.get(node_id, 0) + 1
    for node in supplement:
        degree = degrees.get(node['id'], 0)
        nodes.append(dict(node, color={'background': '#4ade80', 'border': '#4ade80',
            'highlight': {'background': '#ffffff', 'border': '#4ade80'}},
            size=10+min(degree, 20)*.3, font={'size': 0, 'color': '#ffffff'},
            title=f"{node['source_file']}:{node.get('source_location', 'L1')} [AST/integration supplement]",
            community_name='Relocation integration', degree=degree))
    for edge in graph['links']:
        if str(edge['source']).startswith('relocation-index:'):
            edges.append({'from': edge['source'], 'to': edge['target'], 'label': edge['relation'],
                'title': edge['relation'] + ' [AST/integration supplement]',
                'dashes': False, 'width': 1, 'color': {'opacity': .5}})
    array('RAW_NODES', nodes)
    array('RAW_EDGES', edges)
    # Regenerate the legend from displayed data; some old exports leave it empty.
    legend = {}
    for node in nodes:
        cid = node.get('community', 0)
        if cid not in legend:
            color = node.get('color', '#94a3b8')
            legend[cid] = {'cid': cid, 'label': node.get('community_name', f'Community {cid}'),
                'color': color.get('background', '#94a3b8') if isinstance(color, dict) else color, 'count': 0}
        legend[cid]['count'] += 1
    array('LEGEND', list(legend.values()))
    html = re.sub(r'<div id="stats">.*?</div>',
        f'<div id="stats">{len(nodes)} nodes &middot; {len(edges)} edges &middot; {len(legend)} communities</div>', html, count=1)
    path.write_text(html, encoding='utf-8')


def main():
    folder = REPO / 'graphify-out'
    folder.mkdir(exist_ok=True)
    sources = [REPO / 'demo_relocation.py', REPO / 'demo_v1.py',
        REPO / 'demo/seem/tasks/interactive.py', REPO / 'modeling/architectures/seem_model_v1.py']
    sources += sorted((REPO / 'relocation').glob('*.py'))
    index = {'generated_at': datetime.now(timezone.utc).isoformat(),
        'seem_revision': repo_revision(REPO),
        'brushnet_revision': repo_revision(REPO.parent / 'BrushNet'),
        'provenance': 'AST declarations/imports + explicit reviewed integration links; existing Graphify semantic graph retained',
        'files': {}, 'integration': [
            ['relocation/adapters.py', 'demo_v1.py', 'reuses load_model'],
            ['relocation/adapters.py', 'demo/seem/tasks/interactive.py', 'calls interactive_infer_image(return_masks=True)'],
            ['demo/seem/tasks/interactive.py', 'modeling/architectures/seem_model_v1.py', 'evaluate_demo -> prev_mask; negative_stroke'],
            ['relocation/controller.py', 'relocation/worker.py', 'isolated process; request JSON and PNG/NPY artifacts'],
            ['relocation/worker.py', '../BrushNet/src/diffusers/pipelines/brushnet/pipeline_brushnet.py', 'two inpainting calls'],
            ['setup_relocation.sh', 'setup.sh', 'reuses working SEEM setup'],
        ]}
    for file in sources:
        text = file.read_text(encoding='utf-8')
        tree = ast.parse(text)
        symbols = []

        def visit(node, prefix=''):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + child.name
                    symbols.append({'name': name, 'line': child.lineno, 'kind': type(child).__name__})
                    visit(child, name + '.')
                else:
                    visit(child, prefix)

        visit(tree)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append('.' * node.level + (node.module or ''))
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        index['files'][file.relative_to(REPO).as_posix()] = {
            'sha256': hashlib.sha256(text.encode()).hexdigest(), 'symbols': symbols, 'imports': sorted(set(imports))}
    (folder / 'relocation-index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    graph_file = folder / 'graph.json'
    if graph_file.exists():
        graph = json.loads(graph_file.read_text(encoding='utf-8'))
        graph['nodes'] = [n for n in graph['nodes'] if not n['id'].startswith('relocation-index:')]
        graph['links'] = [e for e in graph['links'] if not str(e['source']).startswith('relocation-index:') and not str(e['target']).startswith('relocation-index:')]
        community = max((n.get('community', 0) for n in graph['nodes'] if isinstance(n.get('community', 0), int)), default=0) + 1
        for file, details in index['files'].items():
            parent_id = 'relocation-index:' + file
            graph['nodes'].append({'id': parent_id, 'label': file, 'source_file': file,
                'source_location': 'L1', '_origin': 'ast', 'file_type': 'code', 'community': community})
            for symbol in details['symbols']:
                node_id = parent_id + ':' + symbol['name']
                graph['nodes'].append({'id': node_id, 'label': symbol['name'], 'source_file': file,
                    'source_location': 'L' + str(symbol['line']), '_origin': 'ast', '_callable': True,
                    'file_type': 'code', 'community': community})
                graph['links'].append({'source': parent_id, 'target': node_id, 'relation': 'contains', 'weight': 1, '_origin': 'ast'})
        for source, target, relation in index['integration']:
            source_id = 'relocation-index:' + source
            target_id = 'relocation-index:' + target
            known = {n['id'] for n in graph['nodes']}
            for node_id, file in [(source_id, source), (target_id, target)]:
                if node_id not in known:
                    graph['nodes'].append({'id': node_id, 'label': file, 'source_file': file,
                        'file_type': 'code', 'community': community, '_origin': 'reviewed integration'})
            graph['links'].append({'source': source_id, 'target': target_id,
                'relation': relation, 'weight': 1, '_origin': 'reviewed integration'})
        graph.setdefault('graph', {})['relocation_index'] = 'relocation-index.json'
        graph['graph']['relocation_indexed_at'] = index['generated_at']
        graph_file.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
        extend_html(folder / 'graph.html', graph, community)
    print(f'Indexed {len(index["files"])} files into {folder}')


if __name__ == '__main__':
    main()
