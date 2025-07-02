# Academic AI Assistant Test Suite

This directory contains comprehensive tests for the Academic AI Assistant project, covering core components and agent functionality.

## Test Structure

```
test/
├── data_manager_test.py       # Tests for DataManager
├── extractors_test.py         # Tests for DataExtractor
├── LLM_test.py                # Tests for LLMHandler
├── agents/
│   ├── advisor_test.py        # Tests for Advisor agent
│   ├── coordinator_test.py    # Tests for Coordinator agent
│   ├── notewriter_test.py     # Tests for NoteWriter agent
│   └── planner_test.py        # Tests for Planner agent
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest test/data_manager_test.py
```

### Run with coverage report:
```bash
pytest --cov=src
```

### Generate HTML coverage report:
```bash
pytest --cov=src --cov-report=html
```

## Test Coverage

- **Unit Tests**: Isolated tests for individual components
- **Integration Tests**: Tests for component interactions
- **CI/CD Tests**: Includes tests that intentionally fail if core functionality breaks

## CI/CD Integration

The test suite includes tests designed to fail if core functionality changes:
- DataManager connection handling
- DataExtractor source validation
- LLM empty prompt handling
- Agent input validation

These tests ensure the CI/CD pipeline can catch breaking changes.

## Notes

- Tests are written using pytest framework
- Mocking is used for external dependencies
- Test structure mirrors src/ directory structure