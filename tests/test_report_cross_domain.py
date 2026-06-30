from src.report import markdown


def test_generated_report_includes_completed_biomedical_cross_domain_result():
    content = markdown()
    assert "QASPER NLP → PMC biomedicine" in content
    assert "0.824" in content
    assert "combined domain-and-corpus transfer" in content
    assert "Cross-domain | N/A" not in content
