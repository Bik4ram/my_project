"""add indexing_jobs table for persistent job store

Revision ID: d5b1cc8385d4f2b7
Revises: e28abd752a06b245
Create Date: 2026-08-09 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd5b1cc8385d4f2b7'
down_revision: Union[str, Sequence[str], None] = 'e28abd752a06b245'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'indexing_jobs',
        sa.Column('job_id', sa.String(length=64), nullable=False),
        sa.Column('project_id', sa.String(length=255), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_stage', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('started_at', sa.DateTime(timezone=False), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=False), nullable=True),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column(
            'statistics',
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'),
            nullable=False,
        ),
        sa.Column('root_path', sa.String(), nullable=False, server_default=''),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=False), nullable=False),
        sa.PrimaryKeyConstraint('job_id'),
    )
    op.create_index(
        'ix_indexing_jobs_project_state', 'indexing_jobs', ['project_id', 'state'], unique=False
    )
    op.create_index('ix_indexing_jobs_state', 'indexing_jobs', ['state'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_indexing_jobs_state', table_name='indexing_jobs')
    op.drop_index('ix_indexing_jobs_project_state', table_name='indexing_jobs')
    op.drop_table('indexing_jobs')
