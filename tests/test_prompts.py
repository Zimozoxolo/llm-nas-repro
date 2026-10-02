from llmnas.prompts import parse_children, parse_knowledge_base


def test_parse_json_list():
    txt = '```json\n[{"child_id":"0","operation":"mutation","architecture_code":"(1, 3, 3, 1, 2, 0)","rationale":"x"}]\n```'
    out = parse_children(txt)
    assert out[0]["code"] == (1, 3, 3, 1, 2, 0) and out[0]["operation"] == "mutation"


def test_parse_list_code_and_op_names():
    assert parse_children('[{"architecture_code": [0,1,2,3,4,0]}]')[0]["code"] == (0, 1, 2, 3, 4, 0)
    txt = '[{"architecture_code": "skip_connect, nor_conv_3x3, none, none, avg_pool_3x3, nor_conv_1x1"}]'
    assert parse_children(txt)[0]["code"] == (1, 3, 0, 0, 4, 2)


def test_parse_invalid_and_free_text():
    assert parse_children('[{"architecture_code": "(7, 1, 1, 1, 1, 1)"}]')[0]["code"] is None
    assert parse_children("try (1, 2, 3, 1, 0, 3) and (0,0,0,0,0,1)")[1]["code"] == (0, 0, 0, 0, 0, 1)


def test_parse_kb():
    assert parse_knowledge_base('{"Updated_Knowledge_Base": ["Use a because b"]}') == ["Use a because b"]
    assert parse_knowledge_base('Updated_Knowledge_Base:\n[{"rule_1": "Avoid x because y"}]') == ["Avoid x because y"]
    assert parse_knowledge_base("no json here") is None
