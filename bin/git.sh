#!/bin/bash

# Get the current commit count
commit_count=$(git rev-list --count HEAD)

# Increment the commit count
next_commit_count=$((commit_count + 1))

# Add all changes
git add .

# Check if a custom message is provided
if [ -z "$1" ]; then
  commit_message="auto commit #$next_commit_count"
else
  commit_message="$1 #$next_commit_count"
fi

# Commit with the message
git commit -m "$commit_message"

# Push the changes
git push

# to run ./git_auto.sh "Your custom message"