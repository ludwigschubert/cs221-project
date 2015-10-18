# encoding: UTF-8
# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 20151018222008) do

  # These are extensions that must be enabled in order to support this database
  enable_extension "plpgsql"

  create_table "repositories", force: :cascade do |t|
    t.string   "name"
    t.string   "full_name"
    t.string   "owner"
    t.boolean  "private"
    t.string   "html_url"
    t.string   "description"
    t.boolean  "fork"
    t.string   "url"
    t.string   "forks_url"
    t.string   "keys_url"
    t.string   "collaborators_url"
    t.string   "teams_url"
    t.string   "hooks_url"
    t.string   "issue_events_url"
    t.string   "events_url"
    t.string   "assignees_url"
    t.string   "branches_url"
    t.string   "tags_url"
    t.string   "blobs_url"
    t.string   "git_tags_url"
    t.string   "git_refs_url"
    t.string   "trees_url"
    t.string   "statuses_url"
    t.string   "languages_url"
    t.string   "stargazers_url"
    t.string   "contributors_url"
    t.string   "subscribers_url"
    t.string   "subscription_url"
    t.string   "commits_url"
    t.string   "git_commits_url"
    t.string   "comments_url"
    t.string   "issue_comment_url"
    t.string   "contents_url"
    t.string   "compare_url"
    t.string   "merges_url"
    t.string   "archive_url"
    t.string   "downloads_url"
    t.string   "issues_url"
    t.string   "pulls_url"
    t.string   "milestones_url"
    t.string   "notifications_url"
    t.string   "labels_url"
    t.string   "releases_url"
    t.datetime "pushed_at"
    t.string   "git_url"
    t.string   "ssh_url"
    t.string   "clone_url"
    t.string   "svn_url"
    t.string   "homepage"
    t.integer  "size"
    t.integer  "stargazers_count"
    t.integer  "watchers_count"
    t.string   "language"
    t.boolean  "has_issues"
    t.boolean  "has_downloads"
    t.boolean  "has_wiki"
    t.boolean  "has_pages"
    t.integer  "forks_count"
    t.string   "mirror_url"
    t.integer  "open_issues_count"
    t.integer  "forks"
    t.integer  "open_issues"
    t.integer  "watchers"
    t.string   "default_branch"
    t.float    "score"
    t.datetime "created_at"
    t.datetime "updated_at"
    t.text     "readme_html"
  end

end
