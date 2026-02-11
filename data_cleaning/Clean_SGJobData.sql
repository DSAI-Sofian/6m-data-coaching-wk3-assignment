ALTER TABLE main.SGJobData
DROP COLUMN metadata_expiryDate;

ALTER TABLE main.SGJobData
DROP COLUMN metadata_isPostedOnBehalf;

ALTER TABLE main.SGJobData
DROP COLUMN metadata_jobPostId;

ALTER TABLE main.SGJobData
DROP COLUMN metadata_totalNumberOfView;

ALTER TABLE main.SGJobData
DROP COLUMN occupationId;

ALTER TABLE main.SGJobData
DROP COLUMN status_id;

ALTER TABLE main.SGJobData
ALTER COLUMN categories
TYPE VARCHAR
USING json_extract_string(categories, '$[0].category');

ALTER TABLE main.SGJobData
ADD COLUMN categ VARCHAR;

UPDATE main.SGJobData
SET sector = json_extract_string(categories, '$[0].category');

UPDATE main.SGJobData
SET sector =
  CASE
    WHEN categories IS NULL OR trim(categories) = '' THEN NULL
    ELSE json_extract_string(categories, '$[0].category')
  END;


ALTER TABLE main.SGJobData
DROP COLUMN categories;

COPY main.SGJobData
TO 'output.csv'
(HEADER, DELIMITER ',');

COPY main.SGJobData
TO '\\wsl.localhost\Ubuntu\home\s_sofian\dsai\6m-assignment-1.0\db\output.csv'
(HEADER, DELIMITER ',');

