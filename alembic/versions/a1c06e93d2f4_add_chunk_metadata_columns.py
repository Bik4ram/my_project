"""add RI-06 chunk metadata columns to chunks

Revision ID: a1c06e93d2f4
Revises: d5b1cc8385d4f2b7
Create Date: 2026-08-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1c06e93d2f4'
down_revision: Union[str, Sequence[str], None] = 'd5b1cc8385d4f2b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Additive, nullable columns — existing rows remain readable and are
    # filled on their next ingestion pass. Single source per field; see
    # ingestion/chunking/pipeline.py for where each is set.
    op.add_column('chunks', sa.Column('language', sa.String(length=64), nullable=True))
    op.add_column('chunks', sa.Column('category', sa.String(length=64), nullable=True))
    op.add_column('chunks', sa.Column('parser_name', sa.String(length=64), nullable=True))
    op.add_column('chunks', sa.Column('chunker_name', sa.String(length=64), nullable=True))
    op.add_column('chunks', sa.Column('symbol_name', sa.String(length=256), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chunks', 'symbol_name')
    op.drop_column('chunks', 'chunker_name')
    op.drop_column('chunks', 'parser_name')
    op.drop_column('chunks', 'category')
    op.drop_column('chunks', 'language')