class AddReadmeHtmlToRepositories < ActiveRecord::Migration
  def change
    add_column :repositories, :readme_html, :text
  end
end
