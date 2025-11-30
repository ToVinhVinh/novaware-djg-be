import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models

class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('products', '0002_initial'),
        ('recommendations', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='recommendationrequest',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='recommendation_requests', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='recommendationlog',
            name='request',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='logs', to='recommendations.recommendationrequest'),
        ),
        migrations.AddField(
            model_name='recommendationresult',
            name='products',
            field=models.ManyToManyField(related_name='recommendation_results', to='products.product'),
        ),
        migrations.AddField(
            model_name='recommendationresult',
            name='request',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='result', to='recommendations.recommendationrequest'),
        ),
    ]
