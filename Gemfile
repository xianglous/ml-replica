# frozen_string_literal: true

source "https://rubygems.org"

git_source(:github) {|repo_name| "https://github.com/#{repo_name}" }

# Manage our dependency on the version of the github-pages gem here.
gem "github-pages", "= 226"

# Explicitly include this gem here.
# It is not directly included in the github-pages gem list of dependencies,
# even though it is included in the original GitHub Pages build infrastructure.
gem "jekyll-include-cache", "= 0.2.1"
gem "jekyll-octicons", "~> 14.2"
gem 'wdm', '>= 0.1.0' if Gem.win_platform?
gem "webrick", "~> 1.7"
