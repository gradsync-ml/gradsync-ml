import pytest

from gradsync.pipeline.core import validate_cluster_config


def test_rejects_duplicate_election_nodes():
    with pytest.raises(ValueError, match="duplicate entries in election_nodes"):
        validate_cluster_config(
            ["10.0.0.1:50051", "10.0.0.1:50051"],
            ["10.0.0.1:60051", "10.0.0.2:60051"],
            "10.0.0.1:50051",
            "10.0.0.1:60051",
        )


def test_rejects_duplicate_cluster_nodes():
    with pytest.raises(ValueError, match="duplicate entries in cluster_nodes"):
        validate_cluster_config(
            ["10.0.0.1:50051", "10.0.0.2:50051"],
            ["10.0.0.1:60051", "10.0.0.1:60051"],
            "10.0.0.1:50051",
            "10.0.0.1:60051",
        )


def test_rejects_mismatched_election_and_cluster_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        validate_cluster_config(
            ["10.0.0.1:50051", "10.0.0.2:50051"],
            ["10.0.0.1:60051"],
            "10.0.0.1:50051",
            "10.0.0.1:60051",
        )


def test_rejects_missing_local_election_endpoint():
    with pytest.raises(ValueError, match="local election endpoint"):
        validate_cluster_config(
            ["10.0.0.2:50051", "10.0.0.3:50051"],
            ["10.0.0.1:60051", "10.0.0.2:60051"],
            "10.0.0.1:50051",
            "10.0.0.1:60051",
        )


def test_rejects_missing_local_data_endpoint():
    with pytest.raises(ValueError, match="local data endpoint"):
        validate_cluster_config(
            ["10.0.0.1:50051", "10.0.0.2:50051"],
            ["10.0.0.2:60051", "10.0.0.3:60051"],
            "10.0.0.1:50051",
            "10.0.0.1:60051",
        )


def test_allows_same_ip_with_different_ports():
    validate_cluster_config(
        ["127.0.0.1:51234", "127.0.0.1:51235", "127.0.0.1:51236"],
        ["127.0.0.1:12345", "127.0.0.1:12346", "127.0.0.1:12347"],
        "127.0.0.1:51234",
        "127.0.0.1:12345",
    )
