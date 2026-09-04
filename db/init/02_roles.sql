CREATE ROLE general_role LOGIN PASSWORD 'general_dev_password';

GRANT CONNECT ON DATABASE company_intel TO general_role;
GRANT USAGE ON SCHEMA public TO general_role;
GRANT SELECT ON employees, sales TO general_role;
GRANT SELECT ON employees, sales, lessons_learned TO general_role;

