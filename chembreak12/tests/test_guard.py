import pytest
from chembreak12.guards import RunAccess, validate_run_access

def test_train_is_train_only():
    validate_run_access(RunAccess('train','Train'))
    with pytest.raises(PermissionError): validate_run_access(RunAccess('train','Test1'))

def test_eval_is_tests_only():
    validate_run_access(RunAccess('eval','Test3'))
    with pytest.raises(PermissionError): validate_run_access(RunAccess('eval','Train'))
