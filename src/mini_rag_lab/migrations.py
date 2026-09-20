from pathlib import Path

from psycopg import AsyncConnection


async def apply_migrations(
    database_url: str,
    migration_directory: str | Path = "migrations",
) -> list[str]:
    directory = Path(migration_directory)
    migration_paths = sorted(directory.glob("*.sql"))
    if not migration_paths:
        raise FileNotFoundError(f"no SQL migrations found in {directory}")

    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        applied: list[str] = []
        for path in migration_paths:
            cursor = await connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (path.name,),
            )
            if await cursor.fetchone() is not None:
                continue

            async with connection.transaction():
                await connection.execute(path.read_text(encoding="utf-8"))
                await connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (path.name,),
                )
            applied.append(path.name)

    return applied
