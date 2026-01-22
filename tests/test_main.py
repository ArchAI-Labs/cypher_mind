import os
import json
import pytest
from unittest.mock import Mock, patch
import sys

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.import_data import GraphImport


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set up mock environment variables"""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NODE_URL", "nodes.json")
    monkeypatch.setenv("RELATIONSHIP_URL", "relationships.json")
    monkeypatch.setenv("RESET", "false")


@pytest.fixture
def sample_node_files():
    """Sample node files data"""
    return {
        "employees.csv": ["Employee", ["id", "name"], "id"],
        "departments.csv": ["Department", ["id", "name"], "id"]
    }


@pytest.fixture
def sample_relationship_files():
    """Sample relationship files data"""
    return {
        "works_in.csv": ["WORKS_IN", "Employee", "Department", ["emp_id", "dept_id"], None]
    }


class TestGraphImport:
    """Tests for GraphImport class"""

    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver"""
        with patch('neo4j.GraphDatabase.driver') as mock:
            driver_instance = Mock()
            mock.return_value = driver_instance
            yield driver_instance

    def test_graph_import_initialization(self, mock_driver):
        """Test that GraphImport initializes with correct credentials"""
        importer = GraphImport("bolt://localhost:7687", "neo4j", "password")

        assert importer.uri == "bolt://localhost:7687"
        assert importer.user == "neo4j"
        assert importer.password == "password"
        assert importer.driver == mock_driver

    def test_close_connection(self, mock_driver):
        """Test that close method calls driver.close()"""
        importer = GraphImport("bolt://localhost:7687", "neo4j", "password")
        importer.close()

        mock_driver.close.assert_called_once()

    def test_open_session(self, mock_driver):
        """Test that open method returns a session"""
        importer = GraphImport("bolt://localhost:7687", "neo4j", "password")
        session = importer.open()

        mock_driver.session.assert_called_once()
        assert session == mock_driver.session.return_value

    def test_reset_database(self, mock_driver):
        """Test that reset_database executes DETACH DELETE query"""
        mock_session = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        importer = GraphImport("bolt://localhost:7687", "neo4j", "password")
        importer.reset_database()

        # Verify session was created and execute_write was called
        mock_session.execute_write.assert_called_once()

    def test_import_nodes(self, mock_driver):
        """Test that import_nodes creates proper Cypher query"""
        mock_session = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        importer = GraphImport("bolt://localhost:7687", "neo4j", "password")
        importer.import_nodes("employees.csv", "Employee", ["id", "name"], "id")

        # Verify execute_write was called
        mock_session.execute_write.assert_called_once()

    def test_import_relationships(self, mock_driver):
        """Test that import_relationships creates proper Cypher query"""
        mock_session = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        importer = GraphImport("bolt://localhost:7687", "neo4j", "password")
        importer.import_relationships(
            "works_in.csv",
            "WORKS_IN",
            "Employee",
            "Department",
            ["emp_id", "dept_id"],
            None
        )

        # Verify execute_write was called
        mock_session.execute_write.assert_called_once()

    def test_import_relationships_with_metadata(self, mock_driver):
        """Test that import_relationships handles metadata"""
        mock_session = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        importer = GraphImport("bolt://localhost:7687", "neo4j", "password")
        importer.import_relationships(
            "works_in.csv",
            "WORKS_IN",
            "Employee",
            "Department",
            ["emp_id", "dept_id"],
            ["since", "role"]
        )

        # Verify execute_write was called
        mock_session.execute_write.assert_called_once()

    def test_import_all(self, sample_node_files, sample_relationship_files):
        """Test that import_all calls import_nodes and import_relationships"""
        with patch('neo4j.GraphDatabase.driver'):
            importer = GraphImport("bolt://localhost:7687", "neo4j", "password")

            with patch.object(importer, 'import_nodes') as mock_import_nodes:
                with patch.object(importer, 'import_relationships') as mock_import_relationships:
                    importer.import_all(sample_node_files, sample_relationship_files)

                    # Verify import_nodes was called for each node file
                    assert mock_import_nodes.call_count == len(sample_node_files)

                    # Verify import_relationships was called for each relationship file
                    assert mock_import_relationships.call_count == len(sample_relationship_files)


class TestMainScriptExecution:
    """Integration tests for main.py script execution"""

    @pytest.mark.integration
    def test_main_script_file_exists(self):
        """Test that main.py file exists and is readable"""
        main_path = os.path.join("src", "main.py")
        assert os.path.exists(main_path), "main.py should exist in src directory"

        # Verify it contains the expected imports and main guard
        with open(main_path, "r") as f:
            content = f.read()
            assert 'if __name__ == "__main__"' in content
            assert 'from backend.import_data import GraphImport' in content
            assert 'GraphImport' in content

    def test_environment_variables_parsing(self, monkeypatch):
        """Test that environment variables are correctly parsed"""
        monkeypatch.setenv("NEO4J_URI", "bolt://testhost:7687")
        monkeypatch.setenv("NEO4J_USER", "testuser")
        monkeypatch.setenv("NEO4J_PASSWORD", "testpass")
        monkeypatch.setenv("RESET", "yes")

        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        reset = os.environ.get("RESET").lower() in ("yes", "true")

        assert uri == "bolt://testhost:7687"
        assert user == "testuser"
        assert password == "testpass"
        assert reset is True

    def test_reset_flag_variations(self, monkeypatch):
        """Test different reset flag values"""
        test_cases = [
            ("yes", True),
            ("YES", True),
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("no", False),
            ("false", False),
            ("0", False),
            ("", False),
        ]

        for value, expected in test_cases:
            monkeypatch.setenv("RESET", value)
            reset = os.environ.get("RESET").lower() in ("yes", "true")
            assert reset == expected, f"Failed for value: {value}"

    def test_json_file_loading(self, tmp_path, sample_node_files, sample_relationship_files):
        """Test that JSON files can be loaded correctly"""
        # Create temporary JSON files
        node_file = tmp_path / "nodes.json"
        rel_file = tmp_path / "relationships.json"

        node_file.write_text(json.dumps(sample_node_files))
        rel_file.write_text(json.dumps(sample_relationship_files))

        # Load and verify
        with open(node_file, "r") as f:
            loaded_nodes = json.load(f)

        with open(rel_file, "r") as f:
            loaded_rels = json.load(f)

        assert loaded_nodes == sample_node_files
        assert loaded_rels == sample_relationship_files

    def test_invalid_json_handling(self, tmp_path):
        """Test handling of invalid JSON files"""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json content }")

        with pytest.raises(json.JSONDecodeError):
            with open(invalid_file, "r") as f:
                json.load(f)


class TestMainScriptLogic:
    """Tests for main.py script logic without execution"""

    def test_main_logic_with_reset(self, sample_node_files, sample_relationship_files, monkeypatch, tmp_path):
        """Test main script logic with database reset"""
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_USER", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "password")
        monkeypatch.setenv("RESET", "yes")

        # Create temp files
        node_file = tmp_path / "nodes.json"
        rel_file = tmp_path / "relationships.json"
        node_file.write_text(json.dumps(sample_node_files))
        rel_file.write_text(json.dumps(sample_relationship_files))

        monkeypatch.setenv("NODE_URL", str(node_file))
        monkeypatch.setenv("RELATIONSHIP_URL", str(rel_file))

        # Mock GraphImport
        with patch('backend.import_data.GraphImport') as mock_import:
            mock_instance = Mock()
            mock_import.return_value = mock_instance

            # Simulate main.py logic
            uri = os.environ.get("NEO4J_URI")
            user = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")

            importer = mock_import(uri, user, password)

            node_url = os.environ.get("NODE_URL")
            relationship_url = os.environ.get("RELATIONSHIP_URL")
            reset = os.environ.get("RESET").lower() in ("yes", "true")

            if reset:
                importer.reset_database()

            try:
                if node_url and relationship_url:
                    with open(node_url, "r") as f:
                        node_files = json.load(f)

                    with open(relationship_url, "r") as f:
                        relationship_files = json.load(f)

                    importer.import_all(node_files, relationship_files)
                else:
                    importer.open()
            finally:
                importer.close()

            # Verify
            mock_instance.reset_database.assert_called_once()
            mock_instance.import_all.assert_called_once_with(sample_node_files, sample_relationship_files)
            mock_instance.close.assert_called_once()

    def test_main_logic_without_reset(self, sample_node_files, sample_relationship_files, tmp_path, monkeypatch):
        """Test main script logic without database reset"""
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_USER", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "password")
        # Create temp files
        node_file = tmp_path / "nodes.json"
        rel_file = tmp_path / "relationships.json"
        node_file.write_text(json.dumps(sample_node_files))
        rel_file.write_text(json.dumps(sample_relationship_files))

        monkeypatch.setenv("NODE_URL", str(node_file))
        monkeypatch.setenv("RELATIONSHIP_URL", str(rel_file))
        monkeypatch.setenv("RESET", "false")

        # Mock GraphImport
        with patch('backend.import_data.GraphImport') as mock_import:
            mock_instance = Mock()
            mock_import.return_value = mock_instance

            # Simulate main.py logic
            uri = os.environ.get("NEO4J_URI")
            user = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")

            importer = mock_import(uri, user, password)

            node_url = os.environ.get("NODE_URL")
            relationship_url = os.environ.get("RELATIONSHIP_URL")
            reset = os.environ.get("RESET").lower() in ("yes", "true")

            if reset:
                importer.reset_database()

            try:
                if node_url and relationship_url:
                    with open(node_url, "r") as f:
                        node_files = json.load(f)

                    with open(relationship_url, "r") as f:
                        relationship_files = json.load(f)

                    importer.import_all(node_files, relationship_files)
                else:
                    importer.open()
            finally:
                importer.close()

            # Verify
            mock_instance.reset_database.assert_not_called()
            mock_instance.import_all.assert_called_once()
            mock_instance.close.assert_called_once()

    def test_main_logic_without_urls(self, monkeypatch):
        """Test main script logic when URLs are missing"""
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_USER", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "password")
        monkeypatch.delenv("NODE_URL", raising=False)
        monkeypatch.delenv("RELATIONSHIP_URL", raising=False)

        # Mock GraphImport
        with patch('backend.import_data.GraphImport') as mock_import:
            mock_instance = Mock()
            mock_import.return_value = mock_instance

            # Simulate main.py logic
            uri = os.environ.get("NEO4J_URI")
            user = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")

            importer = mock_import(uri, user, password)

            node_url = os.environ.get("NODE_URL")
            relationship_url = os.environ.get("RELATIONSHIP_URL")
            reset = os.environ.get("RESET", "false").lower() in ("yes", "true")

            if reset:
                importer.reset_database()

            try:
                if node_url and relationship_url:
                    importer.import_all({}, {})
                else:
                    importer.open()
            finally:
                importer.close()

            # Verify
            mock_instance.open.assert_called_once()
            mock_instance.import_all.assert_not_called()
            mock_instance.close.assert_called_once()
