# Need to set ACCESS_TOKEN

namespace :crawl do

  task :ids => :environment do
    Octokit.configure do |c|
      c.access_token = ACCESS_TOKEN
      c.auto_paginate = false
    end
    client = Octokit::Client.new

    while true do
      results = client.all_repositories per_page: 100, since: Repository.maximum(:id)
      # byebug
      repositories = []
      # existing_ids = Repository.pluck(:id)
      results.each do |item|
        item = item.to_hash
        # unless existing_ids.include? item['id'] or existing_ids.include? item[:id]
        repositories << Repository.new(item)
        # end
      end
      Repository.import repositories, validate: false
      puts "Imported repo up to #{Repository.maximum(:id)}."
      puts "Now has #{Repository.count} repos."
    end

  end

  task :metadata => :environment do
    Octokit.configure do |c|
      c.access_token = ACCESS_TOKEN
      c.auto_paginate = false
    end
    client = Octokit::Client.new

    while Repository.where("stargazers_count IS NULL").count > 0 do
      count = Repository.where("stargazers_count IS NULL").count
      puts "Fetching metadata for #{count} more repos..."
      begin
        repository = Repository.where("stargazers_count IS NULL").first!
        result = client.repository repository.id
        hash = result.to_hash.select { |key,_| repository.attributes.keys.include? key.to_s }
        repository.update_attributes( hash )
        puts "Fetched metadata for repo #{repository.id}."
      rescue Exception => e
        repository.destroy
        puts "Exception, keep going."
      end
    end

  end

  task :download => :environment do
    while Repository.where("readme_html IS NULL").count > 0 do
      repository = Repository.where("readme_html IS NULL").first!
      begin
        url = URI.parse repository.html_url
        puts "Downloading #{repository.full_name}"
        full_html = Net::HTTP.get(url)
        repository.readme_html = Nokogiri::HTML(full_html).at_css('#readme')
        repository.save!
	if repository.readme_html.nil?
		repository.destroy
	end
      rescue Timeout::Error, Errno::EINVAL, Errno::ECONNRESET, EOFError,
       Net::HTTPBadResponse, Net::HTTPHeaderSyntaxError, Net::ProtocolError => e
        repository.destroy
        puts "Exception, keep going."
      end
    end
  end

end
