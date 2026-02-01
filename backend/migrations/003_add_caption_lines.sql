-- Add caption_lines field to tasks table
ALTER TABLE tasks ADD COLUMN caption_lines INTEGER DEFAULT 1 CHECK (caption_lines >= 1 AND caption_lines <= 3);

-- Add caption_lines to user preferences (default 1 for single line)
ALTER TABLE users ADD COLUMN default_caption_lines INTEGER DEFAULT 1 CHECK (default_caption_lines >= 1 AND default_caption_lines <= 3);