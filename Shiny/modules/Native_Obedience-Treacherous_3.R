# Virginia_Money-Christ_1.R

library(shiny)
library(ggiraph)
library(dplyr)
library(tidyverse)
library(cluster)
library(factoextra)
library(mclust)
library(stringr)

plot_ui <- function(id) {
  ns <- NS(id)
  girafeOutput(ns("interactive_plot"))
}

plot_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    output$interactive_plot <- renderGirafe({
      # Ensure data is reactive
      df <- data()
      
      # K-means Clustering
      set.seed(27708) # For reproducibility
      
      # Standardization
      df_k <- df %>%
        mutate(
          P2_scaled = scale(P2),
          P2_scaled = scale(P2),
          P2_scaled = scale(P2)
        ) %>%
        select(Filename, P2_scaled, P2_scaled, P2_scaled)
      
      # If anyone were to change the number of clusters for the three clusterings, please change it here:
      k_1 <- 3
      k_2 <- 3
      k_3 <- 3
      
      kmeans_P2 <- kmeans(matrix(df_k$P2_scaled, ncol = 1), centers = k_1, nstart = 25)
      kmeans_P2 <- kmeans(matrix(df_k$P2_scaled, ncol = 1), centers = k_2, nstart = 25)
      kmeans_P2 <- kmeans(matrix(df_k$P2_scaled, ncol = 1), centers = k_3, nstart = 25)
      
      # Add the cluster assignments to the original data frame
      df_clustered_k <- df_k %>% 
        mutate(cluster_P2 = kmeans_P2$cluster,
               cluster_P2 = kmeans_P2$cluster,
               cluster_P2 = kmeans_P2$cluster)
      
      clustered <- df_clustered_k
      
      merged_df <- df %>%
        left_join(clustered, by = "Filename") %>%
        mutate(
          cluster_P2 = as.character(cluster_P2),
          cluster_P2 = as.character(cluster_P2),
          cluster_P2 = as.character(cluster_P2)
        ) %>%
        select(Filename, P2, P2, P2, Title, Author, Year, cluster_P2, cluster_P2, cluster_P2)
      
      # Filter the dataset for the specified filenames
      lines_df <- merged_df %>% filter(Filename %in% c("A73849", "A19313", "A14803"))
      
      annotations <- lines_df %>%
        mutate(label = case_when(
          Filename == "A73849" ~ "J. Donne",
          Filename == "A19313" ~ "P. Copland",
          Filename == "A14803" ~ "E. Waterhouse"
        ),
        x_pos = case_when(
          Filename == "A73849" ~ P2 + 0.08,
          Filename == "A19313" ~ P2 + 0.08,
          Filename == "A14803" ~ P2 - 0.02
        ),
        y_pos = case_when(
          Filename == "A73849" ~ 0.8,
          Filename == "A19313" ~ 1,
          Filename == "A14803" ~ 1
        ))
      
      merged_df$cluster_P2 <- factor(merged_df$cluster_P2, levels = c(1, 3, 2))
      
      # Define custom labels and colors for the clusters
      custom_labels <- c("1" = "Extremely Treacherous", "3" = "Relatively Neutral", "2" = "Obedient-inclined")
      custom_colors <- c("1" = "#66c2a5", "2" = "#fc8d62", "3" = "#8da0cb")
      
      annotated_points <- merged_df %>% filter(Filename %in% c("A73849", "A19313", "A14803"))
      remaining_points <- merged_df %>% filter(!Filename %in% c("A73849", "A19313", "A14803"))
      
      plot <- ggplot() +
        geom_density(data = merged_df, aes(x = P2, color = factor(cluster_P2), fill = factor(cluster_P2)), alpha = 0.05) +  
        geom_jitter_interactive(
          data = remaining_points, 
          aes(x = P2, y = 0, color = factor(cluster_P2), alpha = 0.15, tooltip = paste("Author:", Author, "<br>Filename:", Filename, "<br>Publication:", Year)), 
          width = 0, 
          height = 0.1, 
          size = 1.5
        ) +
        geom_jitter_interactive(
          data = annotated_points, 
          aes(x = P2, y = 0, color = factor(cluster_P2), alpha = 1, tooltip = paste("Author:", Author, "<br>Filename:", Filename, "<br>Publication:", Year)), 
          width = 0, 
          height = 0.1, 
          size = 1.5
        ) +
        geom_vline(
          xintercept = 0, 
          linetype = "dashed", 
          color = "red3", 
          show.legend = FALSE,
          size = 0.7
        ) +
        geom_vline(
          data = lines_df, 
          aes(xintercept = P2, color = factor(cluster_P2)), 
          linetype = "dotted", 
          size = 0.6, 
          show.legend = FALSE
        ) +
        geom_text_interactive(
          data = annotations, 
          aes(x = x_pos, y = y_pos, label = label, color = factor(cluster_P2), tooltip = label), 
          vjust = 1, 
          show.legend = FALSE,
          family = "Roboto",
          fontface = "bold",
          angle = -90
        ) +  
        annotate(
          "text", x = - 0.2, y = Inf, 
          label = "Absolute\nNeutrality", 
          vjust = 1, color = "red3",
          family = "Roboto",
          fontface = "bold"
        ) +
        scale_color_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_fill_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_alpha_identity() +  
        theme_minimal(base_size = 12) +  
        labs(
          title = 'Distribution of Connotations of "Native" \n on Obedience-Treacherous Dichotomy',
          x = 'Note: the more rightward, the more obedient is the connotation of "Native"',
          y = ""
        ) +
        theme(
          axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text.x = element_blank(),
          legend.position = "top",  
          legend.title = element_blank(), 
          legend.text = element_text(
            size = 10, family = "Roboto"
          ),
          panel.grid.major = element_blank(),  
          panel.grid.minor = element_blank(),  
          plot.title = element_text(
            size = 18, hjust = 0.5, margin = margin(b = 20), family = "Roboto",
            face = "bold"
          ),
          text = element_text(family = "Roboto")
        )
      
      girafe(ggobj = plot, width_svg = 7, height_svg = 5)
    })
  })
}
