from parse_policy import parse_policy


def test_parse_policy_yields_six_sections(policy_path):
    sections = parse_policy(policy_path)
    assert len(sections) == 6
    assert [s.section for s in sections] == ["1", "2", "3", "4", "5", "6"]
    assert sections[0].section_title == "Meals"
    assert sections[0].chunk_id == "expense-policy:v2.0:section-1"
    assert sections[0].document == "Employee Expense Policy"
    assert sections[0].version == "2.0"
    assert "65" in sections[0].text
    assert not sections[0].text.startswith("##")
