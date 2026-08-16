"""add area roots to projects

Revision ID: b7c2d3e4f5a6
Revises: a1c06e93d2f4
Create Date: 2026-08-16 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7c2d3e4f5a6'
down_revision: str | Sequence[str] | None = 'a1c06e93d2f4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema.

    Additive, nullable columns — a legacy single-root project keeps
    ``root_path`` populated and all three area roots NULL; a multi-root
    project fills at least one area root. See
    ``core/project_paths.ProjectPaths`` for the resolution semantics.
    """
    op.add_column('projects', sa.Column('rtl_root', sa.Text(), nullable=True))
    op.add_column('projects', sa.Column('tb_root', sa.Text(), nullable=True))
    op.add_column('projects', sa.Column('docs_root', sa.Text(), nullable=True))
    # indexing_jobs mirrors the projects columns so a persisted job can be
    # re-run with the same roots after a worker restart.
    op.add_column('indexing_jobs', sa.Column('rtl_root', sa.String(), nullable=True))
    op.add_column('indexing_jobs', sa.Column('tb_root', sa.String(), nullable=True))
    op.add_column('indexing_jobs', sa.Column('docs_root', sa.String(), nullable=True))
    # chunks gain portable multi-root metadata (additive, nullable).
    op.add_column('chunks', sa.Column('source_type', sa.String(length=16), nullable=True))
    op.add_column('chunks', sa.Column('source_root', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chunks', 'source_root')
    op.drop_column('chunks', 'source_type')
    op.drop_column('indexing_jobs', 'docs_root')
    op.drop_column('indexing_jobs', 'tb_root')
    op.drop_column('indexing_jobs', 'rtl_root')
    op.drop_column('projects', 'docs_root')
    op.drop_column('projects', 'tb_root')
    op.drop_column('projects', 'rtl_root')