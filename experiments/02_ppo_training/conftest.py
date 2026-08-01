import pytest

# Dictionary to store docstrings mappings
test_descriptions = {}

def pytest_collection_modifyitems(session, config, items):
    """Capture the docstring of each test during collection."""
    for item in items:
        doc = item.obj.__doc__
        if doc:
            # Clean up the docstring formatting
            cleaned_doc = " ".join([line.strip() for line in doc.strip().split("\n")])
            test_descriptions[item.nodeid] = cleaned_doc
        else:
            test_descriptions[item.nodeid] = "No description provided."

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Format the final terminal summary for high human-readability."""
    terminalreporter.ensure_newline()
    terminalreporter.section("TITAN V5.0 HUMAN-READABLE TEST SUMMARY", sep="=", blue=True, bold=True)
    
    passed = terminalreporter.stats.get('passed', [])
    failed = terminalreporter.stats.get('failed', [])
    errors = terminalreporter.stats.get('error', [])
    
    all_reports = passed + failed + errors
    
    for rep in all_reports:
        # We only care about the actual test execution phase, unless setup failed
        if rep.when == 'call' or (rep.when == 'setup' and rep.outcome == 'failed'):
            name = rep.nodeid.split("::")[-1]
            desc = test_descriptions.get(rep.nodeid, "No description")
            
            if rep.passed:
                status = "PASSED ✅"
                # TerminalReporter write with colors
                terminalreporter.write(f"[{status}] ", green=True, bold=True)
            else:
                status = "FAILED ❌"
                terminalreporter.write(f"[{status}] ", red=True, bold=True)
                
            terminalreporter.write_line(f"{name}")
            terminalreporter.write_line(f"      Description : {desc}")
            terminalreporter.write_line("")
