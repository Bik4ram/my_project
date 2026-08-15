"""add api_keys table for persistent API key store

Revision ID: e28abd752a06b245
Revises: 0190b83318c7
Create Date: 2026-08-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e28abd752a06b245'
down_revision: Union[str, Sequence[str], None] = '0190b83318c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'api_keys',
        sa.Column('key_hash', sa.String(length=128), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=False), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=False), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='active'),
        sa.Column('description', sa.String(length=1024), nullable=True),
        sa.Column('allowed_projects', sa.JSON(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=False), nullable=True),
        sa.PrimaryKeyConstraint('key_hash'),
    )
    op.create_index('ix_api_keys_status', 'api_keys', ['status'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_api_keys_status', table_name='api_keys')
    op.drop_table('api_keys')
