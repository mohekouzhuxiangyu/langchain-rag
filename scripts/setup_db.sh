#!/usr/bin/env bash
# 初始化 PostgreSQL + pgvector 数据库（macOS/Homebrew 环境）
# 说明：brew 的 pgvector 预编译包支持 postgresql@17/18，因此本项目使用 PG17。
set -euo pipefail

PG_VER="${PG_VER:-17}"
DB_NAME="${PG_DATABASE:-media_rag}"
PG_USER="${PG_USER:-postgres}"
PG_PASSWORD="${PG_PASSWORD:-postgres}"

echo "==> 检查 PostgreSQL@${PG_VER}..."
if ! command -v psql >/dev/null 2>&1; then
  echo "psql 未安装，执行: brew install postgresql@${PG_VER} pgvector"
  exit 1
fi

DATA_DIR="$(brew --prefix)/var/postgresql@${PG_VER}"
if [ ! -d "$DATA_DIR" ]; then
  echo "==> 初始化数据目录..."
  initdb --locale=C -E UTF-8 "$DATA_DIR" >/dev/null
fi

echo "==> 启动 PostgreSQL..."
if ! pg_isready -q 2>/dev/null; then
  brew services start postgresql@${PG_VER} || pg_ctl -D "$DATA_DIR" -l /tmp/pg.log start
  sleep 2
fi

echo "==> 确保 pgvector 扩展对 PG${PG_VER} 可用..."
# brew pgvector 若未包含当前 PG 版本，则从源码编译（需网络可访问 GitHub）
if [ ! -f "$(brew --prefix)/share/postgresql@${PG_VER}/extension/vector.control" ]; then
  echo "    pgvector 缺少 PG${PG_VER} 版本，尝试源码编译..."
  cd /tmp && rm -rf pgvector && git clone --depth 1 https://github.com/pgvector/pgvector.git
  cd pgvector && make PG_CONFIG="$(brew --prefix)/opt/postgresql@${PG_VER}/bin/pg_config" && make install
fi

echo "==> 设置 postgres 密码..."
psql -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='${PG_USER}'" | grep -q 1 || \
  psql -d postgres -c "CREATE ROLE ${PG_USER} WITH LOGIN SUPERUSER PASSWORD '${PG_PASSWORD}';"
psql -d postgres -c "ALTER ROLE ${PG_USER} WITH PASSWORD '${PG_PASSWORD}';"

echo "==> 创建数据库 ${DB_NAME}..."
psql -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
  psql -d postgres -c "CREATE DATABASE ${DB_NAME} OWNER ${PG_USER};"

echo "==> 启用 pgvector..."
psql -U "${PG_USER}" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "完成。数据库连接: postgresql://${PG_USER}:***@127.0.0.1:5432/${DB_NAME}"
