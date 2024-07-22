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
          P1_scaled = scale(P1),
          P2_scaled = scale(P2),
          P3_scaled = scale(P3)
        ) %>%
        select(Filename, P1_scaled, P2_scaled, P3_scaled)
      
      # If anyone were to change the number of clusters for the three clusterings, please change it here:
      k_1 <- 3
      k_2 <- 3
      k_3 <- 4
      
      kmeans_P1 <- kmeans(matrix(df_k$P1_scaled, ncol = 1), centers = k_1, nstart = 25)
      kmeans_P2 <- kmeans(matrix(df_k$P2_scaled, ncol = 1), centers = k_2, nstart = 25)
      kmeans_P3 <- kmeans(matrix(df_k$P3_scaled, ncol = 1), centers = k_3, nstart = 25)
      
      # Add the cluster assignments to the original data frame
      df_clustered_k <- df_k %>% 
        mutate(cluster_P1 = kmeans_P1$cluster,
               cluster_P2 = kmeans_P2$cluster,
               cluster_P3 = kmeans_P3$cluster)
      
      clustered <- df_clustered_k
      
      merged_df <- df %>%
        left_join(clustered, by = "Filename") %>%
        mutate(
          cluster_P1 = as.character(cluster_P1),
          cluster_P2 = as.character(cluster_P2),
          cluster_P3 = as.character(cluster_P3)
        ) %>%
        select(Filename, P1, P2, P3, Title, Author, Year, cluster_P1, cluster_P2, cluster_P3)
      
      # Filter the dataset for the specified filenames
      lines_df <- merged_df %>% filter(Filename %in% c("A73849", "A19313", "A14803"))
      
      annotations <- lines_df %>%
        mutate(label = case_when(
          Filename == "A73849" ~ "J. Donne",
          Filename == "A19313" ~ "P. Copland",
          Filename == "A14803" ~ "E. Waterhouse"
        ),
        x_pos = case_when(
          Filename == "A73849" ~ P1 - 0.02,
          Filename == "A19313" ~ P1 + 0.08,
          Filename == "A14803" ~ P1 + 0.08
        ),
        y_pos = case_when(
          Filename == "A73849" ~ 1.3,
          Filename == "A19313" ~ 0.7,
          Filename == "A14803" ~ 1
        ))
      
      merged_df$cluster_P1 <- factor(merged_df$cluster_P1, levels = c(3, 2, 1))
      
      # Define custom labels and colors for the clusters
      custom_labels <- c("3" = "More Religious-driven", "2" = "Relatively Neutral", "1" = "Very Financial-driven")
      custom_colors <- c("1" = "#66c2a5", "2" = "#fc8d62", "3" = "#8da0cb")
      
      annotated_points <- merged_df %>% filter(Filename %in% c("A73849", "A19313", "A14803"))
      remaining_points <- merged_df %>% filter(!Filename %in% c("A73849", "A19313", "A14803"))
      
      plot <- ggplot() +
        geom_density(data = merged_df, aes(x = P1, color = factor(cluster_P1), fill = factor(cluster_P1)), alpha = 0.05) +  
        geom_jitter_interactive(
          data = remaining_points, 
          aes(x = P1, y = 0, color = factor(cluster_P1), alpha = 0.15, tooltip = paste("Author:", Author, "<br>Filename:", Filename, "<br>Publication:", Year)), 
          width = 0, 
          height = 0.1, 
          size = 1.5
        ) +
        geom_jitter_interactive(
          data = annotated_points, 
          aes(x = P1, y = 0, color = factor(cluster_P1), alpha = 1, tooltip = paste("Author:", Author, "<br>Filename:", Filename, "<br>Publication:", Year)), 
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
          aes(xintercept = P1, color = factor(cluster_P1)), 
          linetype = "dotted", 
          size = 0.6, 
          show.legend = FALSE
        ) +
        geom_text_interactive(
          data = annotations, 
          aes(x = x_pos, y = y_pos, label = label, color = factor(cluster_P1), tooltip = label), 
          vjust = 1, 
          show.legend = FALSE,
          family = "Roboto",
          fontface = "bold",
          angle = 0
        ) +  
        annotate(
          "text", x = -0.25, y = Inf, 
          label = "Absolute\n Neutrality", 
          vjust = 1, color = "red3",
          family = "Roboto",
          fontface = "bold"
        ) +
        scale_color_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_fill_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_alpha_identity() +  
        scale_x_continuous(limits = c(-0.3, 3.2)) +
        theme_minimal(base_size = 12) +  
        labs(
          title = 'Distribution of Connotations of "Virginia" \n on Money-Christ Dichotomy',
          x = 'Note: the more rightward, the more monetary is the connotation of "Virginia"',
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
          plot.margin = unit(c(1, 1, 1, 1), "cm"),
          text = element_text(family = "Roboto")
        )
      
      girafe(ggobj = plot, width_svg = 7, height_svg = 5)
    })
  })
}
