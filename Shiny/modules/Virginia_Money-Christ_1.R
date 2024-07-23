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
  tagList(
    girafeOutput(ns("interactive_plot")),
    girafeOutput(ns("line_plot"))
  )
}

plot_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    
    # Define custom labels and colors for the clusters
    custom_labels <- c("3" = "More Religious-driven", "2" = "Relatively Neutral", "1" = "Very Financial-driven")
    custom_colors <- c("1" = "#66c2a5", "2" = "#fc8d62", "3" = "#8da0cb")
    
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
      k_3 <- 3
      
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
          Filename == "A19313" ~ P1 + 0.12,
          Filename == "A14803" ~ P1 + 0.12
        ),
        y_pos = case_when(
          Filename == "A73849" ~ 1.3,
          Filename == "A19313" ~ 0.7,
          Filename == "A14803" ~ 1
        ))
      
      merged_df$cluster_P1 <- factor(merged_df$cluster_P1, levels = c(3, 2, 1))
      
      
      
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
          family = "Open Sans",
          fontface = "bold",
          angle = -90
        ) +  
        annotate(
          "text", x = -0.25, y = Inf, 
          label = "Absolute\n Neutrality", 
          vjust = 1, color = "red3",
          family = "Open Sans",
          fontface = "bold",
          angle = 0,
          size = 3
        ) +
        scale_color_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_fill_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_alpha_identity() +  
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
            size = 10, family = "Open Sans"
          ),
          panel.grid.major = element_blank(),  
          panel.grid.minor = element_blank(),  
          plot.title = element_text(
            size = 18, hjust = 0.5, margin = margin(b = 20), family = "Open Sans",
            face = "bold"
          ),
          plot.margin = unit(c(1, 1, 1, 1), "cm"),
          text = element_text(family = "Open Sans")
        )
      
      girafe(ggobj = plot, width_svg = 7, height_svg = 5)
    })
    
    output$line_plot <- renderGirafe({
      df <- data()
      
      process_year <- function(year) {
        # Remove square brackets
        year <- str_replace_all(year, "\\[|\\]", "")
        
        # Split by comma if there are multiple years
        years <- str_split(year, ",\\s*")[[1]]
        # Convert to numeric and take the minimum if there are multiple years
        if (length(years) > 1) {
          year <- min(as.numeric(years))
        }
        return(as.character(year))
      }
      
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
      k_3 <- 3
      
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
      
      
      cy_df <- merged_df %>%
        mutate(Year = as.numeric(sapply(Year, process_year))) %>%
        filter(Year < 1626 & Year >= 1606) %>%
        mutate(Year_Interval = cut(Year, breaks = seq(1606, 1627, by = 5), right = FALSE, labels = seq(1606, 1622, by = 5)))
      
      percentage_p1 <- cy_df %>%
        group_by(Year_Interval) %>%
        mutate(total_works = n()) %>%
        group_by(Year_Interval, cluster_P1) %>%
        summarise(count = n(), total_works = first(total_works)) %>%
        mutate(percentage = (count / total_works) * 100) %>%
        ungroup()
      
      percentage_interval <- percentage_p1 %>%
        mutate(new_x = case_when(
          Year_Interval == "1606" ~ "1606 - 1610",
          Year_Interval == "1611" ~ "1611 - 1615",
          Year_Interval == "1616" ~ "1616 - 1620",
          Year_Interval == "1621" ~ "1621 - 1625"
        ))
      
      percentage_interval <- percentage_interval %>%
        mutate(tooltip_text = paste("Number of works:", count, "\nTime Frame:", new_x))
      
      line_plot <- ggplot(percentage_interval, aes(x = Year_Interval, y = percentage, color = as.factor(cluster_P1), group = cluster_P1)) +
        geom_line(size = 0.8) +
        geom_point_interactive(aes(tooltip = tooltip_text), size = 1.5) +
        geom_text(
          aes(
            label = sprintf("%.1f%%", percentage)
          ),
          vjust = -1, hjust = 0.5,
          size = 3,
          family = "Open Sans",
          show.legend = FALSE,
          fontface = "bold"
        ) +
        scale_color_manual(values = custom_colors, labels = custom_labels, name = "") +  
        labs(title = 'Attitudes about "Virginia" Change over Time\n(5Y Average)',
             x = "Year Interval",
             y = "Percentage") +
        scale_y_continuous(expand = expansion(mult = c(0, 0.1)), limits = c(0, NA)) +  
        theme_minimal(base_size = 12) +
        theme(
          plot.title = element_text(
            size = 18, family = "Open Sans", face = "bold", 
            margin = margin(b = 20),
            hjust = 0.5
          ),
          axis.title = element_text(size = 12, family = "Open Sans"),
          axis.text = element_text(size = 12, family = "Open Sans"),
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          legend.title = element_text(size = 10, family = "Open Sans"),
          legend.text = element_text(size = 10, family = "Open Sans"),
          legend.position = "top",
          panel.grid.major = element_line(color = "grey80", size = 0.25),
          panel.grid.minor = element_blank(),
          plot.margin = unit(c(1, 1, 1, 1), "cm")
        )
      
      girafe(ggobj = line_plot, width_svg = 7, height_svg = 5)
    })
  })
}
