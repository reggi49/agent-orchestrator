import os
import stat
from pathlib import Path

import pytest

from agent_system.file_writer import ControlledFileWriter, UnapprovedFileError


def setup_project(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "backend" / "test.go").write_text("package test\n")
    return p


def test_18_approved_file_written(tmp_path):
    """18. Approved file dapat ditulis."""
    p = setup_project(tmp_path)
    writer = ControlledFileWriter(p, ["backend/test.go"], "run1")
    target = writer.write_file("backend/test.go", "package test\n// updated\n")
    assert target.is_file()
    assert "// updated" in target.read_text()


def test_19_unapproved_file_rejected(tmp_path):
    """19. File di luar proposal ditolak."""
    p = setup_project(tmp_path)
    writer = ControlledFileWriter(p, ["backend/test.go"], "run1")
    with pytest.raises(UnapprovedFileError):
        writer.write_file("backend/other.go", "package other\n")


def test_20_backup_created(tmp_path):
    """20. Backup dibuat sebelum file ditimpa."""
    p = setup_project(tmp_path)
    (p / "backend" / "test.go").write_text("original content")
    writer = ControlledFileWriter(p, ["backend/test.go"], "run1")
    writer.write_file("backend/test.go", "modified content")

    backup_file = p / ".agent" / "runtime" / "backups" / "run1" / "backend" / "test.go"
    assert backup_file.is_file()
    assert backup_file.read_text() == "original content"


def test_21_atomic_write_used(tmp_path):
    """21. Atomic write digunakan (tidak meninggalkan file corrupt)."""
    p = setup_project(tmp_path)
    writer = ControlledFileWriter(p, ["backend/test.go"], "run1")
    target = writer.write_file("backend/test.go", "atomic content")
    assert target.read_text() == "atomic content"
    # Ensure tmp directory does not keep the tmp file
    tmp_files = list((p / ".agent" / "tmp").glob("tmp_write_*"))
    assert len(tmp_files) == 0


def test_22_file_permissions_preserved(tmp_path):
    """22. File permission dipertahankan."""
    p = setup_project(tmp_path)
    target_f = p / "backend" / "test.go"
    target_f.chmod(0o755)

    writer = ControlledFileWriter(p, ["backend/test.go"], "run1")
    writer.write_file("backend/test.go", "new content")

    mode = stat.S_IMODE(target_f.stat().st_mode)
    assert mode == 0o755


def test_23_symlink_escaping_rejected(tmp_path):
    """23. Symlink keluar project ditolak."""
    p = setup_project(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    sym = p / "backend" / "sym.go"
    try:
        os.symlink(outside, sym)
    except OSError:
        pytest.skip("Symlink not supported")

    writer = ControlledFileWriter(p, ["backend/sym.go"], "run1")
    with pytest.raises(ValueError):
        writer.write_file("backend/sym.go", "hack")


def test_24_absolute_path_rejected(tmp_path):
    """24. Absolute path ditolak."""
    p = setup_project(tmp_path)
    writer = ControlledFileWriter(p, ["backend/test.go"], "run1")
    with pytest.raises(ValueError):
        writer.write_file("/etc/hosts", "hack")


def test_25_protocol_file_rejected(tmp_path):
    """25. Protocol file ditolak."""
    p = setup_project(tmp_path)
    writer = ControlledFileWriter(p, [".agent/manifest.yaml"], "run1")
    with pytest.raises(ValueError):
        writer.write_file(".agent/manifest.yaml", "hack")


def test_26_secret_file_rejected(tmp_path):
    """26. Secret file ditolak."""
    p = setup_project(tmp_path)
    writer = ControlledFileWriter(p, ["backend/.env"], "run1")
    with pytest.raises(ValueError):
        writer.write_file("backend/.env", "SECRET=1")
